from .runtime import ROOT, load_config, parse_args, load_state_dict
import argparse
import os
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from .chicago import (
    build_sensor_sequences,
    chicago_include_precipitation_input,
    chicago_split_indices,
    chicago_use_terrain_conditioning,
    extend_chicago_sensor_map,
    fit_chicago_normalizers,
    get_chicago_output_size,
    get_chicago_state_array,
    get_chicago_target_normalizer,
    load_chicago_manifest,
    load_chicago_sensors,
    resolve_chicago_observables,
    resolve_chicago_targets,
)
from .chicago_forecast import build_chicago_feature_coords
from .evaluate import get_model_params
from .models import CSTRIDE
from .processdata import Padding, num2p


def seed_latent_training(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class LatentPredictionLSTM(torch.nn.Module):
    def __init__(self, latent_size, hidden_size, num_layers, dropout):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = torch.nn.LSTM(
            input_size=latent_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.head = torch.nn.Linear(hidden_size, latent_size)

    def forward(self, latent_history):
        output, _ = self.lstm(latent_history)
        return self.head(output[:, -1])


def load_state_dict_compatible(path, device):
    state_dict = load_state_dict(path, device)
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k[len("_orig_mod.") :]: v for k, v in state_dict.items()}
    return state_dict


def split_indices_for_name(split_name, entries, split_file=None):
    idx_train, idx_valid, idx_test = chicago_split_indices(entries, split_file)
    if split_name == "train":
        return idx_train
    if split_name == "valid":
        return idx_valid
    if split_name == "test":
        return idx_test
    raise ValueError(f"Unsupported split: {split_name}")


def finite_mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr))


def safe_relative_error(pred, target):
    denom = np.sum(target**2)
    if denom <= 0:
        return float("nan")
    return float(np.sqrt(np.sum((pred - target) ** 2) / denom))


def num2p_safe(value):
    return num2p(value) if np.isfinite(value) else "nan%"


def plot_step_errors(step_mean, step_std, output_path, annotate_steps=(1, 4, 6, 12, 24)):
    steps = np.arange(1, len(step_mean) + 1)
    mean = np.asarray(step_mean, dtype=np.float64)
    std = np.asarray(step_std, dtype=np.float64)

    plt.figure(figsize=(6, 3))
    plt.plot(steps, mean, color="#1f77b4", linewidth=2.0, marker="o", markersize=3, label="Mean")
    plt.fill_between(steps, mean - std, mean + std, color="#1f77b4", alpha=0.22, label="Std")
    for step in annotate_steps:
        if 1 <= step <= len(mean) and np.isfinite(mean[step - 1]):
            plt.scatter([step], [mean[step - 1]], color="#d62728", s=28, zorder=3)
            plt.annotate(
                num2p_safe(mean[step - 1]),
                (step, mean[step - 1]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                va="bottom",
                fontsize=8,
                color="#d62728",
            )
    plt.xlabel("Forecast step")
    plt.ylabel("Relative error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def compute_observed_steps(ntimes, observed_fraction):
    if observed_fraction >= 1.0:
        return ntimes
    return min(max(1, int(np.floor(ntimes * observed_fraction))), ntimes)


def encode_latents(model, histories):
    h_0 = torch.zeros((model.hidden_layers, histories.size(0), model.hidden_size), device=histories.device)
    c_0 = torch.zeros((model.hidden_layers, histories.size(0), model.hidden_size), device=histories.device)
    _, (latent, _) = model.lstm(histories, (h_0, c_0))
    return latent[-1].view(-1, model.hidden_size)


def extract_latent_sequences(model, sensors_data, trajectory_indices, steps, cstride_lag, device, batch_size):
    model.eval()
    latent_sequences = {}
    with torch.no_grad():
        for idx in trajectory_indices:
            idx = int(idx)
            histories = Padding(sensors_data[idx, :steps].unsqueeze(0), cstride_lag)[:steps]
            latents = []
            for start in range(0, histories.shape[0], batch_size):
                batch = histories[start : start + batch_size].to(device)
                latents.append(encode_latents(model, batch).cpu())
            latent_sequences[idx] = torch.cat(latents, dim=0)
    return latent_sequences


def make_latent_prediction_tensors(latent_sequences, trajectory_indices, training_steps, latent_lag):
    histories = []
    targets = []
    for idx in trajectory_indices:
        latent_seq = latent_sequences[int(idx)][:training_steps].float()
        padded = Padding(latent_seq.unsqueeze(0), latent_lag)[:training_steps]
        histories.append(padded[:-1])
        targets.append(latent_seq[1:])
    return torch.cat(histories, dim=0), torch.cat(targets, dim=0)


def train_latent_model(model, train_loader, valid_loader, args, device, ckpt_path):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.latent_lr, weight_decay=args.latent_weight_decay)
    criterion = torch.nn.MSELoss()
    best_val = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.latent_epochs + 1):
        model.train()
        train_loss = 0.0
        for history, target in train_loader:
            history = history.to(device)
            target = target.to(device)
            pred = model(history)
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            if args.latent_grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.latent_grad_clip)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for history, target in valid_loader:
                history = history.to(device)
                target = target.to(device)
                val_loss += criterion(model(history), target).item()
        val_loss /= max(1, len(valid_loader))

        if epoch == 1 or epoch % args.latent_log_interval == 0:
            print(f"LatentPred Epoch {epoch}/{args.latent_epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        if val_loss < best_val - args.latent_improvement_threshold:
            best_val = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "latent_size": args._latent_size,
                    "hidden_size": args.latent_hidden_size,
                    "num_layers": args.latent_layers,
                    "dropout": args.latent_dropout,
                    "best_val_loss": best_val,
                    "best_epoch": best_epoch,
                },
                ckpt_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= args.latent_patience:
                print(f"LatentPred early stopping at epoch {epoch}.")
                break

    checkpoint = load_state_dict(ckpt_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return float(checkpoint["best_val_loss"]), int(checkpoint["best_epoch"])


def sample_start_steps(eval_indices, ntimes, horizon, min_start, num_starts, seed):
    max_start = ntimes - horizon
    if max_start < min_start:
        raise ValueError(
            f"No valid evaluation start steps for ntimes={ntimes}, horizon={horizon}, min_start={min_start}."
        )
    candidates = np.arange(min_start, max_start + 1, dtype=int)
    if candidates.shape[0] < num_starts:
        raise ValueError(
            f"Requested {num_starts} random starts per trajectory, but only {candidates.shape[0]} are available."
        )
    rng = np.random.default_rng(seed)
    return {int(idx): np.sort(rng.choice(candidates, size=num_starts, replace=False)).astype(int) for idx in eval_indices}


def build_static_rollout_tensors(entry, root, start_step, eval_horizon, normalizers, target_indices, target_normalizer, terrain_conditioning):
    coords = np.load(root / entry["coords_file"], mmap_mode="r")
    if terrain_conditioning:
        elevation = np.load(root / entry["elevation_file"], mmap_mode="r")
        manning = np.load(root / entry["manning_file"], mmap_mode="r")
        slope = np.load(root / entry["slope_file"], mmap_mode="r")
    else:
        elevation = None
        manning = None
        slope = None

    feature_coords = build_chicago_feature_coords(
        coords,
        elevation,
        manning,
        slope,
        normalizers,
        terrain_conditioning=terrain_conditioning,
    )
    future_coords = np.repeat(feature_coords[None, None, :, :], eval_horizon, axis=1)

    state = get_chicago_state_array(root, entry)
    target = state[start_step : start_step + eval_horizon, :, :][:, :, target_indices]
    target = target_normalizer.transform(target).astype(np.float32)[None]
    return torch.tensor(future_coords, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


def autoregressive_forecast_with_latent_predictions(cstride_model, latent_model, latent_history, batch_coords):
    predictions = []
    horizon = batch_coords.shape[1]
    for step in range(horizon):
        next_latent = latent_model(latent_history)
        prediction = cstride_model.decode(batch_coords[:, step], next_latent)
        predictions.append(prediction)
        latent_history = torch.cat((latent_history[:, 1:], next_latent.unsqueeze(1)), dim=1)
    return torch.stack(predictions, dim=1)


def append_results(
    run_path,
    split_name,
    observed_fraction,
    observed_steps,
    eval_horizon,
    include_precipitation,
    terrain_conditioning,
    latent_best_epoch,
    latent_best_val,
    inference_time,
    cstride_total_params,
    cstride_trainable_params,
    latent_total_params,
    latent_trainable_params,
    latent_seed,
    latent_lag,
    eval_seed,
    num_starts,
    start_steps_by_entry,
    step_mean,
    step_std,
    plot_path,
):
    with open(os.path.join(run_path, "results.txt"), "a") as f:
        f.write("\nChicago Reconstruction-Backbone Latent-Prediction Evaluation\n")
        f.write("============================================================\n")
        f.write(f"Split: {split_name}\n")
        f.write(f"Observed time fraction: {observed_fraction}\n")
        f.write(f"Observed training steps: {observed_steps}\n")
        f.write(f"Autoregressive horizon: {eval_horizon}\n")
        f.write(f"Evaluation seed: {eval_seed}\n")
        f.write(f"Random valid starts per trajectory: {num_starts}\n")
        for entry_idx, starts in start_steps_by_entry.items():
            f.write(f"Evaluation start steps for trajectory index {entry_idx}: {starts.tolist()}\n")
        f.write(f"include_precipitation_input: {include_precipitation}\n")
        f.write(f"terrain_conditioning: {terrain_conditioning}\n")
        f.write(f"Latent LSTM best epoch: {latent_best_epoch}\n")
        f.write(f"Latent LSTM best validation loss: {latent_best_val}\n")
        f.write(f"C-STRIDE total parameters: {cstride_total_params}\n")
        f.write(f"C-STRIDE trainable parameters: {cstride_trainable_params}\n")
        f.write(f"Latent LSTM total parameters: {latent_total_params}\n")
        f.write(f"Latent LSTM trainable parameters: {latent_trainable_params}\n")
        f.write(f"Latent LSTM training seed: {latent_seed}\n")
        f.write(f"Latent LSTM lag: {latent_lag}\n")
        f.write(f"Inference time: {inference_time:.2f} seconds\n")
        f.write(f"Relative error plot: {plot_path}\n")
        for step_idx, (mean, std) in enumerate(zip(step_mean, step_std), start=1):
            f.write(
                f"Forecast step {step_idx} denormalized reconstruction relative error with predicted latents: "
                f"mean={mean} ({num2p_safe(mean)}), std={std}\n"
            )


def main(args):
    run_path = Path(args.path)
    with open(run_path / "config.yaml", "r") as f:
        config = load_config(f)
    if args.latent_seed is None:
        args.latent_seed = int(config["basics"].get("random_seed", 0))
    if args.eval_seed is None:
        args.eval_seed = int(config["basics"].get("random_seed", 0))
    seed_latent_training(args.latent_seed)
    if config["basics"].get("scaler", "global") != "global":
        raise ValueError("Chicago reconstruction-backbone latent prediction currently supports only global scaling.")

    root, manifest, entries = load_chicago_manifest(config["basics"].get("data_root", "data/Chicago"))
    ntimes = int(manifest["time_steps"])
    observed_fraction = float(config.get("forecast", {}).get("observed_time_fraction", 1.0))
    observed_steps = compute_observed_steps(ntimes, observed_fraction)
    eval_horizon = int(args.horizon)
    if eval_horizon <= 0:
        raise ValueError(f"Evaluation horizon must be positive, got {eval_horizon}.")
    if observed_steps < 2:
        raise ValueError(f"Latent LSTM training requires at least 2 observed steps, got {observed_steps}.")

    cstride_lag = int(config["basics"]["lag"])
    latent_lag = int(args.latent_lag) if args.latent_lag is not None else cstride_lag
    dataset_output_size = get_chicago_output_size(root, entries[0])
    observable_indices = resolve_chicago_observables(config, dataset_output_size)
    target_indices = resolve_chicago_targets(config, dataset_output_size)
    output_size = len(target_indices)
    include_precipitation = chicago_include_precipitation_input(config)
    terrain_conditioning = chicago_use_terrain_conditioning(config)
    coord_size = 5 if terrain_conditioning else 2

    idx_train, idx_valid, _ = chicago_split_indices(entries, config["basics"].get("split_file"))
    eval_indices = split_indices_for_name(args.split, entries, config["basics"].get("split_file"))
    min_eval_start = max(cstride_lag, latent_lag, 1)
    start_steps_by_entry = sample_start_steps(
        eval_indices,
        ntimes,
        eval_horizon,
        min_eval_start,
        int(args.num_starts),
        int(args.eval_seed),
    )
    normalizers = fit_chicago_normalizers(entries, idx_train, root)
    target_normalizer = get_chicago_target_normalizer(normalizers, target_indices)
    sensor_map = load_chicago_sensors(str(run_path), config)
    if sensor_map is None:
        raise FileNotFoundError(f"No saved Chicago sensor mapping found in {run_path}.")
    sensor_map = extend_chicago_sensor_map(entries, root, sensor_map)

    sensor_sequences = build_sensor_sequences(
        entries,
        root,
        sensor_map,
        normalizers,
        observable_indices,
        include_precipitation=include_precipitation,
    )
    sensors_data = torch.stack(sensor_sequences, dim=0)
    input_size = sensors_data.shape[-1]

    device = torch.device(config["basics"]["device"] if torch.cuda.is_available() else "cpu")
    cstride_model = CSTRIDE(
        input_size=input_size,
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
    ckpt_path = run_path / "best_model.ckpt"
    if not ckpt_path.exists():
        ckpt_path = run_path / "final_model.ckpt"
    cstride_model.load_state_dict(load_state_dict_compatible(ckpt_path, device))
    cstride_model.eval()

    latent_indices = np.unique(np.concatenate([idx_train, idx_valid, eval_indices])).astype(int)
    latent_sequences = extract_latent_sequences(
        cstride_model,
        sensors_data,
        latent_indices,
        ntimes,
        cstride_lag,
        device,
        args.latent_extract_batch_size,
    )

    train_x, train_y = make_latent_prediction_tensors(latent_sequences, idx_train, observed_steps, latent_lag)
    valid_x, valid_y = make_latent_prediction_tensors(
        latent_sequences,
        idx_valid if len(idx_valid) > 0 else idx_train,
        observed_steps,
        latent_lag,
    )
    train_loader_generator = torch.Generator()
    train_loader_generator.manual_seed(args.latent_seed)
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.latent_batch_size,
        shuffle=True,
        generator=train_loader_generator,
    )
    valid_loader = DataLoader(TensorDataset(valid_x, valid_y), batch_size=args.latent_batch_size, shuffle=False)

    latent_size = cstride_model.hidden_size
    latent_model = LatentPredictionLSTM(
        latent_size=latent_size,
        hidden_size=args.latent_hidden_size,
        num_layers=args.latent_layers,
        dropout=args.latent_dropout,
    ).to(device)
    args._latent_size = latent_size
    latent_ckpt_path = run_path / args.latent_ckpt_name
    latent_best_val, latent_best_epoch = train_latent_model(latent_model, train_loader, valid_loader, args, device, latent_ckpt_path)
    latent_model.eval()

    cstride_total_params, cstride_trainable_params = get_model_params(cstride_model)
    latent_total_params, latent_trainable_params = get_model_params(latent_model)
    step_sample_rel = [[] for _ in range(eval_horizon)]

    start = time.time()
    with torch.no_grad():
        for entry_idx in eval_indices:
            entry_idx = int(entry_idx)
            entry = entries[entry_idx]
            for start_step in start_steps_by_entry[entry_idx]:
                latent_history = Padding(latent_sequences[entry_idx][:start_step].unsqueeze(0), latent_lag)[
                    start_step - 1
                ].unsqueeze(0)
                batch_coords, batch_target = build_static_rollout_tensors(
                    entry,
                    root,
                    start_step,
                    eval_horizon,
                    normalizers,
                    target_indices,
                    target_normalizer,
                    terrain_conditioning,
                )

                latent_history = latent_history.to(device)
                batch_coords = batch_coords.to(device)
                batch_target = batch_target.to(device)

                pred = autoregressive_forecast_with_latent_predictions(
                    cstride_model,
                    latent_model,
                    latent_history,
                    batch_coords,
                )

                pred_np = pred.cpu().numpy()
                target_np = batch_target.cpu().numpy()
                pred_denorm = target_normalizer.inverse_transform(pred_np.reshape(-1, output_size)).reshape(pred_np.shape)
                target_denorm = target_normalizer.inverse_transform(target_np.reshape(-1, output_size)).reshape(target_np.shape)
                for step_idx in range(eval_horizon):
                    step_sample_rel[step_idx].append(
                        safe_relative_error(pred_denorm[0, step_idx], target_denorm[0, step_idx])
                    )
    inference_time = time.time() - start

    step_mean = []
    step_std = []
    for values in step_sample_rel:
        mean, std = finite_mean_std(values)
        step_mean.append(mean)
        step_std.append(std)

    plot_name = args.output if args.output is not None else f"reconstruction_latentpred_relative_error_{args.split}_h{eval_horizon}.png"
    plot_path = run_path / plot_name
    plot_step_errors(step_mean, step_std, plot_path)

    append_results(
        str(run_path),
        args.split,
        observed_fraction,
        observed_steps,
        eval_horizon,
        include_precipitation,
        terrain_conditioning,
        latent_best_epoch,
        latent_best_val,
        inference_time,
        cstride_total_params,
        cstride_trainable_params,
        latent_total_params,
        latent_trainable_params,
        args.latent_seed,
        latent_lag,
        args.eval_seed,
        args.num_starts,
        start_steps_by_entry,
        step_mean,
        step_std,
        str(plot_path),
    )

    print(f"C-STRIDE model loaded from {ckpt_path}")
    print(f"Latent LSTM checkpoint saved to {latent_ckpt_path}")
    print(f"Latent LSTM training seed: {args.latent_seed}")
    print(f"Observed training steps: {observed_steps}/{ntimes} (observed_time_fraction={observed_fraction})")
    print(f"Autoregressive horizon: {eval_horizon}")
    print(f"Evaluation seed: {args.eval_seed}")
    print(f"Evaluation starts per trajectory: {args.num_starts}; total samples: {sum(len(v) for v in start_steps_by_entry.values())}")
    print(f"Relative error plot saved to {plot_path}")
    for step_idx, (mean, std) in enumerate(zip(step_mean, step_std), start=1):
        print(f"Forecast step {step_idx}: denormalized relative error mean={mean}, std={std}")


if __name__ == "__main__":
    torch.set_default_device("cpu")
    parser = argparse.ArgumentParser(description="Forecast Chicago fields using an auxiliary latent LSTM and a reconstruction backbone")
    parser.add_argument("--path", required=True, help="Saved C-STRIDE run directory")
    parser.add_argument("--horizon", type=int, default=24, help="Autoregressive rollout horizon")
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--output", default=None, help="Plot filename relative to the run directory")
    parser.add_argument("--latent_ckpt_name", default="latent_prediction_lstm.ckpt")
    parser.add_argument("--latent_lag", type=int, default=None, help="Latent LSTM input lag; defaults to basics.lag")
    parser.add_argument("--latent_hidden_size", type=int, default=128)
    parser.add_argument("--latent_layers", type=int, default=2)
    parser.add_argument("--latent_dropout", type=float, default=0.0)
    parser.add_argument("--latent_epochs", type=int, default=1000)
    parser.add_argument("--latent_batch_size", type=int, default=256)
    parser.add_argument("--latent_lr", type=float, default=1e-3)
    parser.add_argument("--latent_weight_decay", type=float, default=1e-4)
    parser.add_argument("--latent_patience", type=int, default=50)
    parser.add_argument("--latent_improvement_threshold", type=float, default=1e-5)
    parser.add_argument("--latent_grad_clip", type=float, default=1.0)
    parser.add_argument("--latent_log_interval", type=int, default=10)
    parser.add_argument("--latent_extract_batch_size", type=int, default=256)
    parser.add_argument("--latent_seed", type=int, default=None, help="Latent LSTM seed; defaults to basics.random_seed")
    parser.add_argument("--eval_seed", type=int, default=None, help="Random seed for evaluation start-step sampling; defaults to basics.random_seed")
    parser.add_argument("--num_starts", type=int, default=10, help="Random valid start steps per evaluation trajectory")
    main(parse_args(parser))
