from .runtime import ROOT, load_config, parse_args, load_state_dict
import argparse
import os
import random
from time import time

import numpy as np
import torch
from .tracking import wandb
import yaml
from torch.utils.data import DataLoader

from .soap import SOAP
from .chicago import (
    add_chicago_input_noise,
    build_chicago_sensor_map,
    build_sensor_sequences,
    chicago_include_precipitation_input,
    chicago_split_indices,
    chicago_use_terrain_conditioning,
    compute_chicago_spatial_probs,
    fit_chicago_normalizers,
    get_chicago_output_size,
    load_chicago_manifest,
    load_chicago_sensors,
    resolve_chicago_observables,
    resolve_chicago_targets,
    save_chicago_sensors,
)
from .chicago_forecast import (
    ChicagoForecastWindowDataset,
    make_forecast_histories,
    make_future_exogenous,
)
from .forecast import autoregressive_forecast, discounted_forecast_loss
from .models import CSTRIDEForecast
from .train import save_training_results


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def truncate_for_observed_fraction(tensor, observed_fraction, horizon, split_name):
    if observed_fraction >= 1.0:
        return tensor, tensor.shape[1]
    keep_steps = max(horizon + 1, int(np.floor(tensor.shape[1] * observed_fraction)))
    keep_steps = min(keep_steps, tensor.shape[1])
    print(f"Using first {keep_steps}/{tensor.shape[1]} time steps for {split_name}.")
    return tensor[:, :keep_steps], keep_steps


def observable_output_positions(observable_indices, target_indices):
    missing = [idx for idx in observable_indices if idx not in target_indices]
    if missing:
        raise ValueError(f"Observed channels {missing} are not included in target_variables={target_indices}.")
    return [target_indices.index(idx) for idx in observable_indices]


def run_epoch(model, dataloader, optimizer, criterion, device, config, epoch, training):
    model.train(training)
    total_loss = 0.0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_idx, (batch_x, batch_coords, batch_target, batch_sensor_coords, batch_future_forcing) in enumerate(dataloader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_coords = batch_coords.to(device, non_blocking=True)
            batch_target = batch_target.to(device, non_blocking=True)
            batch_sensor_coords = batch_sensor_coords.to(device, non_blocking=True)
            batch_future_forcing = batch_future_forcing.to(device, non_blocking=True)

            predictions = autoregressive_forecast(
                model=model,
                batch_x=batch_x,
                batch_coords=batch_coords,
                future_sensor_coords=batch_sensor_coords,
                observables=config["_runtime"]["observable_output_positions"],
                future_exogenous=batch_future_forcing,
            )
            loss = discounted_forecast_loss(
                predictions,
                batch_target,
                config["forecast"]["discount_factor"],
                criterion,
            )

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            if training and batch_idx % config["training"]["log_interval"] == 0:
                print(f"Train Epoch: {epoch} [{batch_idx}/{len(dataloader)}]\tLoss: {loss.item():.6f}")

    avg_loss = total_loss / max(1, len(dataloader))
    print(f"====> {'Train' if training else 'Valid'} Loss: {avg_loss:.6f}")
    return avg_loss


def train_forecast_model(model, train_loader, valid_loader, optimizer, scheduler, criterion, device, config, run_dir):
    best_val_loss = float("inf")
    patience_counter = 0
    eval_interval = int(config["training"].get("eval_interval", 10))
    patience_steps = max(1, config["training"]["patience"] // eval_interval)
    threshold = float(config["training"].get("improvement_threshold", 5e-3))
    start = time()

    for epoch in range(1, config["training"]["epochs"] + 1):
        print("-" * 30)
        print(f"Epoch {epoch}/{config['training']['epochs']}")
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device, config, epoch, True)
        log_payload = {"epoch": epoch, "epoch_train_loss": train_loss}

        if epoch % eval_interval == 0:
            val_loss = run_epoch(model, valid_loader, optimizer, criterion, device, config, epoch, False)
            scheduler.step(val_loss)
            log_payload["epoch_val_loss"] = val_loss
            if val_loss < best_val_loss:
                torch.save(model.state_dict(), os.path.join(run_dir, "best_model.ckpt"))
                print(f"Best model saved at epoch {epoch}: {val_loss:.6f}")
            if val_loss < best_val_loss * (1 - threshold):
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience_steps:
                    print("Early stopping triggered.")
                    break
            best_val_loss = min(best_val_loss, val_loss)
        else:
            scheduler.step(train_loss)

        wandb.log(log_payload)
        print(f"====> Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

    training_time = time() - start
    torch.save(model.state_dict(), os.path.join(run_dir, "final_model.ckpt"))
    return training_time


def main(config):
    seed_everything(int(config["basics"]["random_seed"]))
    if config["basics"].get("scaler", "global") != "global":
        raise ValueError("Chicago forecast currently supports only global scaling.")
    if int(config["forecast"].get("lookback_steps", 0)) != 0:
        raise ValueError("Chicago forecast does not currently support lookback_steps > 0.")

    root, manifest, entries = load_chicago_manifest(config["basics"].get("data_root", "data/Chicago"))
    ntimes = int(manifest["time_steps"])
    horizon = int(config["forecast"]["horizon"])
    lag = int(config["basics"]["lag"])
    dataset_output_size = get_chicago_output_size(root, entries[0])
    observable_indices = resolve_chicago_observables(config, dataset_output_size)
    target_indices = resolve_chicago_targets(config, dataset_output_size)
    config["_runtime"] = {
        "observable_output_positions": observable_output_positions(observable_indices, target_indices)
    }
    output_size = len(target_indices)
    num_sensors = int(config["basics"].get("num_sensors", 0))
    include_precipitation = chicago_include_precipitation_input(config)
    terrain_conditioning = chicago_use_terrain_conditioning(config)
    if num_sensors == 0 and not include_precipitation:
        raise ValueError(
            "Chicago forecast training requires at least one temporal input feature. "
            "Set num_sensors > 0 or include_precipitation_input: true."
        )
    coord_size = 5 if terrain_conditioning else 2
    idx_train, idx_valid, _ = chicago_split_indices(entries, config["basics"].get("split_file"))
    normalizers = fit_chicago_normalizers(entries, idx_train, root)

    sensor_map = load_chicago_sensors("", config)
    if sensor_map is None:
        sensor_map = build_chicago_sensor_map(entries, idx_train, root, config)

    sensor_sequences = build_sensor_sequences(
        entries,
        root,
        sensor_map,
        normalizers,
        observable_indices,
        include_precipitation=include_precipitation,
    )
    sensors_data = torch.stack(sensor_sequences, dim=0)
    sensor_feature_count = num_sensors * len(observable_indices)
    if config["basics"]["noise"] > 0.0:
        sensors_data = add_chicago_input_noise(
            sensors_data,
            config["basics"]["noise"],
            sensor_feature_count=sensor_feature_count,
        )

    print(
        f"Loaded Chicago forecast dataset with {len(entries)} trajectories, {ntimes} time steps, "
        f"dataset_output_size={dataset_output_size}, target_indices={target_indices}, "
        f"output_size={output_size}, num_sensors={num_sensors}, observables={observable_indices}, "
        f"include_precipitation_input={include_precipitation}, "
        f"terrain_conditioning={terrain_conditioning}"
    )

    observed_fraction = float(config["forecast"].get("observed_time_fraction", 1.0))
    train_sensors, train_steps = truncate_for_observed_fraction(sensors_data[idx_train], observed_fraction, horizon, "train")
    valid_sensors, valid_steps = truncate_for_observed_fraction(sensors_data[idx_valid], observed_fraction, horizon, "valid")

    train_entries = [entries[i] for i in idx_train]
    valid_entries = [entries[i] for i in idx_valid]
    train_windows = make_forecast_histories(train_sensors, lag, horizon)
    valid_windows = make_forecast_histories(valid_sensors, lag, horizon)
    train_future_forcing = make_future_exogenous(
        train_entries,
        root,
        normalizers,
        horizon,
        max_steps=train_steps,
        include_precipitation=include_precipitation,
    )
    valid_future_forcing = make_future_exogenous(
        valid_entries,
        root,
        normalizers,
        horizon,
        max_steps=valid_steps,
        include_precipitation=include_precipitation,
    )

    spatial_probs_train = {}
    spatial_probs_valid = {}
    if config["training"]["num_points"] is not None and config["training"].get("importance_sampling", False):
        spatial_probs_train = compute_chicago_spatial_probs(entries, idx_train, root, lag)
        spatial_probs_valid = compute_chicago_spatial_probs(entries, idx_valid, root, lag)

    train_dataset = ChicagoForecastWindowDataset(
        train_entries,
        root,
        train_windows,
        train_future_forcing,
        train_steps - horizon,
        horizon,
        normalizers,
        target_indices,
        sensor_map,
        terrain_conditioning=terrain_conditioning,
        num_points=config["training"]["num_points"],
        spatial_probs=spatial_probs_train,
    )
    valid_dataset = ChicagoForecastWindowDataset(
        valid_entries,
        root,
        valid_windows,
        valid_future_forcing,
        valid_steps - horizon,
        horizon,
        normalizers,
        target_indices,
        sensor_map,
        terrain_conditioning=terrain_conditioning,
        num_points=config["training"]["num_points"],
        spatial_probs=spatial_probs_valid,
    )

    batch_size = config["training"]["batch_size"]
    if config["training"]["num_points"] is None and batch_size > 1:
        print("Chicago data uses variable grids; forcing batch_size=1 when num_points is None.")
        batch_size = 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config["training"].get("num_workers", 8), pin_memory=True, persistent_workers=config["training"].get("num_workers", 8) > 0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=config["training"].get("num_workers", 8), pin_memory=True, persistent_workers=config["training"].get("num_workers", 8) > 0)

    device = torch.device(config["basics"]["device"] if torch.cuda.is_available() else "cpu")
    model = CSTRIDEForecast(
        input_size=train_dataset.x_data.shape[-1],
        hidden_size=config["model"]["hidden_size"],
        hidden_layers=config["model"]["hidden_layers"],
        fmmnn_depth=config["model"]["fmmnn_depth"],
        fmmnn_width=config["model"]["fmmnn_width"],
        fmmnn_rank=config["model"]["fmmnn_rank"],
        coord_size=coord_size,
        output_size=output_size,
        num_fourier=config["model"]["num_fourier"],
        device=device,
    ).to(device)
    model = torch.compile(model) if config["training"].get("compile", False) else model

    run_name = f"training_{int(time())}"
    run_dir = os.path.join(config["training"]["save_path"], run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump({k: v for k, v in config.items() if k != "_runtime"}, f, default_flow_style=False)
    save_chicago_sensors(config["training"]["save_path"], run_name, sensor_map)

    wandb.init( project=config.get("wandb", {}).get("project", "C-STRIDE"), config=config, name=run_name)
    wandb.watch(model, log="gradients", log_freq=500)

    optimizer = SOAP(model.parameters(), lr=config["training"]["learning_rate"], betas=(0.95, 0.95), weight_decay=0.01, precondition_frequency=10)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=config["training"]["gamma"], patience=4, threshold=0.01, threshold_mode="rel", min_lr=1e-6)
    criterion = torch.nn.MSELoss(reduction="mean")
    training_time = train_forecast_model(model, train_loader, valid_loader, optimizer, scheduler, criterion, device, config, run_dir)
    save_training_results(config["training"]["save_path"], run_name, training_time)


if __name__ == "__main__":
    torch.set_default_device("cpu")
    parser = argparse.ArgumentParser(description="Train C-STRIDE autoregressive model on the Chicago dataset")
    parser.add_argument("--config", default=str(ROOT / "configs/forecast/autoregressive.yaml"))
    args = parse_args(parser)
    with open(args.config, "r") as f:
        main(load_config(f))
