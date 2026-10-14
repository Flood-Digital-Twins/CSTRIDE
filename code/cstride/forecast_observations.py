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
    chicago_event_seed_indices,
    chicago_use_terrain_conditioning,
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


def seed_observation_training(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class SensorObservationLSTM(torch.nn.Module):
    def __init__(self, input_size, sensor_size, hidden_size, num_layers, dropout):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.head = torch.nn.Linear(hidden_size, sensor_size)

    def forward(self, history):
        output, _ = self.lstm(history)
        return self.head(output[:, -1])


def load_state_dict_compatible(path, device):
    state_dict = load_state_dict(path, device)
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k[len("_orig_mod.") :]: v for k, v in state_dict.items()}
    return state_dict


def observable_output_positions(observable_indices, target_indices):
    missing = [idx for idx in observable_indices if idx not in target_indices]
    if missing:
        raise ValueError(f"Observed channels {missing} are not included in target_variables={target_indices}.")
    return [target_indices.index(idx) for idx in observable_indices]


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




def precip_mismatch_slug(precip_mismatch):
    if float(precip_mismatch) == 0.0:
        return ""
    text = f"{float(precip_mismatch):g}"
    text = text.replace("-", "m").replace(".", "p")
    return f"_precip_mismatch_{text}"


def precip_mismatch_header(precip_mismatch):
    if float(precip_mismatch) == 0.0:
        return ""
    return " With Precipitation Mismatch"

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


def make_observation_prediction_tensors(
    sensors_data,
    trajectory_indices,
    training_steps,
    lag,
    sensor_feature_count,
    input_feature_count,
    forecast_horizon=1,
):
    histories = []
    targets = []
    future_inputs = []
    for idx in trajectory_indices:
        # Only use the observed training prefix. The unobserved suffix is reserved for autoregressive evaluation.
        feature_seq = sensors_data[int(idx), :training_steps, :input_feature_count].float()
        sensor_seq = feature_seq[:, :sensor_feature_count]
        padded = Padding(feature_seq.unsqueeze(0), lag)[:training_steps]
        sample_count = training_steps - forecast_horizon
        histories.append(padded[:sample_count])
        targets.append(torch.stack([sensor_seq[step + 1 : step + 1 + forecast_horizon] for step in range(sample_count)], dim=0))
        future_inputs.append(
            torch.stack(
                [feature_seq[step + 1 : step + 1 + forecast_horizon, sensor_feature_count:input_feature_count] for step in range(sample_count)],
                dim=0,
            )
        )
    histories = torch.cat(histories, dim=0)
    targets = torch.cat(targets, dim=0)
    future_inputs = torch.cat(future_inputs, dim=0)
    if forecast_horizon == 1:
        targets = targets[:, 0]
    return histories, targets, future_inputs


def rollout_observation_model(model, history, horizon, future_inputs):
    predictions = []
    current_history = history
    for step in range(horizon):
        next_obs = model(current_history)
        predictions.append(next_obs)
        if current_history.shape[-1] > next_obs.shape[-1]:
            next_features = torch.cat((next_obs, future_inputs[:, step]), dim=-1)
        else:
            next_features = next_obs
        current_history = torch.cat((current_history[:, 1:], next_features.unsqueeze(1)), dim=1)
    return torch.stack(predictions, dim=1)


def current_multistep_horizon(args, epoch):
    if not args.obs_multistep_loss:
        return 1
    if not args.obs_curriculum:
        return args.obs_multistep_horizon
    return min(args.obs_multistep_horizon, 1 + (epoch - 1) // args.obs_curriculum_epochs)


def train_observation_model(model, train_loader, valid_loader, args, device, ckpt_path):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.obs_lr, weight_decay=args.obs_weight_decay)
    criterion = torch.nn.MSELoss()
    best_val = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.obs_epochs + 1):
        model.train()
        train_loss = 0.0
        for history, target, future_inputs in train_loader:
            history = history.to(device)
            target = target.to(device)
            future_inputs = future_inputs.to(device)
            if args.obs_multistep_loss:
                horizon = current_multistep_horizon(args, epoch)
                pred = rollout_observation_model(model, history, horizon, future_inputs[:, :horizon])
                loss = criterion(pred, target[:, :horizon])
            else:
                pred = model(history)
                loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            if args.obs_grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.obs_grad_clip)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for history, target, future_inputs in valid_loader:
                history = history.to(device)
                target = target.to(device)
                future_inputs = future_inputs.to(device)
                if args.obs_multistep_loss:
                    horizon = current_multistep_horizon(args, epoch)
                    val_loss += criterion(rollout_observation_model(model, history, horizon, future_inputs[:, :horizon]), target[:, :horizon]).item()
                else:
                    val_loss += criterion(model(history), target).item()
        val_loss /= max(1, len(valid_loader))

        if epoch == 1 or epoch % args.obs_log_interval == 0:
            print(f"ObsPred Epoch {epoch}/{args.obs_epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        if val_loss < best_val - args.obs_improvement_threshold:
            best_val = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": args._input_size,
                    "sensor_size": args._sensor_size,
                    "hidden_size": args.obs_hidden_size,
                    "num_layers": args.obs_layers,
                    "dropout": args.obs_dropout,
                    "best_val_loss": best_val,
                    "best_epoch": best_epoch,
                    "multistep_loss": args.obs_multistep_loss,
                    "multistep_horizon": args.obs_multistep_horizon if args.obs_multistep_loss else 1,
                    "curriculum": args.obs_curriculum,
                    "curriculum_epochs": args.obs_curriculum_epochs,
                },
                ckpt_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= args.obs_patience:
                print(f"ObsPred early stopping at epoch {epoch}.")
                break

    checkpoint = load_state_dict(ckpt_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return float(checkpoint["best_val_loss"]), int(checkpoint["best_epoch"])

def load_observation_model_checkpoint(model, args, device, ckpt_path):
    checkpoint = load_state_dict(ckpt_path, device)
    expected = {
        "input_size": args._input_size,
        "sensor_size": args._sensor_size,
        "hidden_size": args.obs_hidden_size,
        "num_layers": args.obs_layers,
        "dropout": args.obs_dropout,
    }
    mismatches = []
    for key, expected_value in expected.items():
        actual_value = checkpoint.get(key)
        if actual_value != expected_value:
            mismatches.append(f"{key}: checkpoint={actual_value}, current={expected_value}")
    if mismatches:
        raise ValueError(
            f"Existing observation LSTM checkpoint {ckpt_path} is incompatible with the current settings: "
            + "; ".join(mismatches)
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    return float(checkpoint.get("best_val_loss", float("nan"))), int(checkpoint.get("best_epoch", -1))


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




def make_precip_mismatch_seed(base_seed, entry_idx, start_step):
    return (int(base_seed) + 1_000_003 * int(entry_idx) + 9_176 * int(start_step)) % (2**63 - 1)


def impose_precip_mismatch(future_precipitation, precip_mismatch, seed):
    if future_precipitation.shape[-1] == 0 or precip_mismatch <= 0.0:
        return future_precipitation
    generator = torch.Generator(device=future_precipitation.device)
    generator.manual_seed(int(seed) % (2**63 - 1))
    noise = torch.randn(
        future_precipitation.shape,
        dtype=future_precipitation.dtype,
        device=future_precipitation.device,
        generator=generator,
    )
    horizon = future_precipitation.shape[1]
    growth = torch.arange(1, horizon + 1, dtype=future_precipitation.dtype, device=future_precipitation.device)
    growth = growth.view(1, horizon, 1) / float(horizon)
    step_std = torch.std(future_precipitation, dim=-1, keepdim=True, unbiased=False)
    perturbed = future_precipitation + float(precip_mismatch) * growth * step_std * noise
    return torch.clamp(perturbed, -1.0, 1.0)

def make_future_precip(entry, root, normalizers, start_step, eval_horizon, include_precipitation):
    if not include_precipitation:
        return torch.empty((1, eval_horizon, 0), dtype=torch.float32)
    precip = np.load(root / entry["precip_file"], mmap_mode="r")
    precip = normalizers["precip"].transform(precip)
    return torch.tensor(precip[start_step : start_step + eval_horizon][None], dtype=torch.float32)


def autoregressive_forecast_with_observation_predictions(
    cstride_model,
    obs_model,
    cstride_history,
    obs_history,
    batch_coords,
    future_precipitation,
    obs_future_precipitation,
):
    predictions = []
    predicted_observations = []
    horizon = batch_coords.shape[1]
    for step in range(horizon):
        prediction, _ = cstride_model(cstride_history, batch_coords[:, step])
        predictions.append(prediction)

        next_sensor_obs = obs_model(obs_history)
        predicted_observations.append(next_sensor_obs)
        if future_precipitation.shape[-1] > 0:
            cstride_next_features = torch.cat((next_sensor_obs, future_precipitation[:, step]), dim=-1)
        else:
            cstride_next_features = next_sensor_obs
        if obs_future_precipitation.shape[-1] > 0:
            obs_next_features = torch.cat((next_sensor_obs, obs_future_precipitation[:, step]), dim=-1)
        else:
            obs_next_features = next_sensor_obs
        cstride_history = torch.cat((cstride_history[:, 1:], cstride_next_features.unsqueeze(1)), dim=1)
        obs_history = torch.cat((obs_history[:, 1:], obs_next_features.unsqueeze(1)), dim=1)
    return torch.stack(predictions, dim=1), torch.stack(predicted_observations, dim=1)


def append_results(
    run_path,
    split_name,
    observed_fraction,
    observed_steps,
    eval_horizon,
    include_precipitation,
    obs_include_precipitation,
    precip_mismatch,
    terrain_conditioning,
    obs_best_epoch,
    obs_best_val,
    inference_time,
    cstride_total_params,
    cstride_trainable_params,
    obs_total_params,
    obs_trainable_params,
    step_mean,
    step_std,
    obs_step_mean,
    obs_step_std,
    obs_seed,
    obs_lag,
    obs_lr,
    obs_weight_decay,
    obs_batch_size,
    obs_grad_clip,
    obs_multistep_loss,
    obs_multistep_horizon,
    obs_curriculum,
    obs_curriculum_epochs,
    obs_ckpt_loaded,
    eval_seed,
    num_starts,
    start_steps_by_entry,
    plot_path,
):
    with open(os.path.join(run_path, "results.txt"), "a") as f:
        title = f"Chicago Reconstruction-Backbone Observation-Prediction Evaluation{precip_mismatch_header(precip_mismatch)}"
        f.write(f"\n{title}\n")
        f.write(f"{'=' * len(title)}\n")
        f.write(f"Split: {split_name}\n")
        f.write(f"Observed time fraction: {observed_fraction}\n")
        f.write(f"Observed training steps: {observed_steps}\n")
        f.write(f"Autoregressive horizon: {eval_horizon}\n")
        f.write(f"Evaluation seed: {eval_seed}\n")
        f.write(f"Random valid starts per trajectory: {num_starts}\n")
        for entry_idx, starts in start_steps_by_entry.items():
            f.write(f"Evaluation start steps for trajectory index {entry_idx}: {starts.tolist()}\n")
        f.write(f"include_precipitation_input: {include_precipitation}\n")
        f.write(f"Observation LSTM include_precipitation_input: {obs_include_precipitation}\n")
        f.write(f"Precipitation mismatch: {precip_mismatch}\n")
        f.write("Precipitation mismatch scale: per-step true normalized precipitation std with linear rollout growth\n")
        f.write(f"terrain_conditioning: {terrain_conditioning}\n")
        f.write(f"Observation LSTM best epoch: {obs_best_epoch}\n")
        f.write(f"Observation LSTM best validation loss: {obs_best_val}\n")
        f.write(f"Observation LSTM checkpoint loaded: {obs_ckpt_loaded}\n")
        f.write(f"C-STRIDE total parameters: {cstride_total_params}\n")
        f.write(f"C-STRIDE trainable parameters: {cstride_trainable_params}\n")
        f.write(f"Observation LSTM total parameters: {obs_total_params}\n")
        f.write(f"Observation LSTM trainable parameters: {obs_trainable_params}\n")
        f.write(f"Observation LSTM training seed: {obs_seed}\n")
        f.write(f"Observation LSTM lag: {obs_lag}\n")
        f.write(f"Observation LSTM learning rate: {obs_lr}\n")
        f.write(f"Observation LSTM weight decay: {obs_weight_decay}\n")
        f.write(f"Observation LSTM batch size: {obs_batch_size}\n")
        f.write(f"Observation LSTM grad clip: {obs_grad_clip}\n")
        f.write(f"Observation LSTM multi-step loss: {obs_multistep_loss}\n")
        f.write(f"Observation LSTM multi-step horizon: {obs_multistep_horizon}\n")
        f.write(f"Observation LSTM curriculum: {obs_curriculum}\n")
        f.write(f"Observation LSTM curriculum epochs: {obs_curriculum_epochs}\n")
        f.write(f"Inference time: {inference_time:.2f} seconds\n")
        f.write(f"Relative error plot: {plot_path}\n")
        for step_idx, (mean, std) in enumerate(zip(step_mean, step_std), start=1):
            f.write(
                f"Forecast step {step_idx} denormalized reconstruction relative error with predicted observations: "
                f"mean={mean} ({num2p_safe(mean)}), std={std}\n"
            )
        for step_idx, (mean, std) in enumerate(zip(obs_step_mean, obs_step_std), start=1):
            f.write(
                f"Forecast step {step_idx} denormalized observation relative error: "
                f"mean={mean} ({num2p_safe(mean)}), std={std}\n"
            )


def main(args):
    run_path = Path(args.path)
    with open(run_path / "config.yaml", "r") as f:
        config = load_config(f)
    if args.obs_seed is None:
        args.obs_seed = int(config["basics"].get("random_seed", 0))
    if args.eval_seed is None:
        args.eval_seed = int(config["basics"].get("random_seed", 0))
    if args.precip_mismatch < 0.0:
        raise ValueError(f"precip_mismatch must be nonnegative, got {args.precip_mismatch}.")
    seed_observation_training(args.obs_seed)
    if config["basics"].get("scaler", "global") != "global":
        raise ValueError("Chicago reconstruction-backbone observation prediction currently supports only global scaling.")

    root, manifest, entries = load_chicago_manifest(config["basics"].get("data_root", "data/Chicago"))
    ntimes = int(manifest["time_steps"])
    observed_fraction = float(config.get("forecast", {}).get("observed_time_fraction", 1.0))
    observed_steps = compute_observed_steps(ntimes, observed_fraction)
    eval_horizon = int(args.horizon)
    if eval_horizon <= 0:
        raise ValueError(f"Evaluation horizon must be positive, got {eval_horizon}.")

    cstride_lag = int(config["basics"]["lag"])
    obs_lag = int(args.obs_lag) if args.obs_lag is not None else cstride_lag
    obs_training_steps = observed_steps
    dataset_output_size = get_chicago_output_size(root, entries[0])
    observable_indices = resolve_chicago_observables(config, dataset_output_size)
    target_indices = resolve_chicago_targets(config, dataset_output_size)
    output_size = len(target_indices)
    include_precipitation = chicago_include_precipitation_input(config)
    terrain_conditioning = chicago_use_terrain_conditioning(config)
    obs_include_precipitation = bool(args.obs_include_precipitation_input)
    if args.obs_ckpt_name is None:
        args.obs_ckpt_name = "sensor_observation_lstm_precip.ckpt" if obs_include_precipitation else "sensor_observation_lstm.ckpt"
    coord_size = 5 if terrain_conditioning else 2

    idx_train, idx_valid, _ = chicago_split_indices(entries, config["basics"].get("split_file"))
    eval_indices = split_indices_for_name(args.split, entries, config["basics"].get("split_file"))
    seed_indices = chicago_event_seed_indices(entries, config["basics"].get("split_file"))
    min_eval_start = max(cstride_lag, obs_lag, 1)
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
    observation_normalizer = get_chicago_target_normalizer(normalizers, observable_indices)
    sensor_map = load_chicago_sensors(str(run_path), config)
    if sensor_map is None:
        raise FileNotFoundError(f"No saved Chicago sensor mapping found in {run_path}.")

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
    precip_size = int(np.asarray(normalizers["precip"].min_value).size) if include_precipitation else 0
    sensor_feature_count = input_size - precip_size
    if obs_include_precipitation == include_precipitation:
        obs_sensors_data = sensors_data
    else:
        obs_sensor_sequences = build_sensor_sequences(
            entries,
            root,
            sensor_map,
            normalizers,
            observable_indices,
            include_precipitation=obs_include_precipitation,
        )
        obs_sensors_data = torch.stack(obs_sensor_sequences, dim=0)
    obs_input_size = obs_sensors_data.shape[-1]
    obs_precip_size = int(np.asarray(normalizers["precip"].min_value).size) if obs_include_precipitation else 0
    obs_sensor_feature_count = obs_input_size - obs_precip_size
    if args.obs_multistep_horizon <= 0:
        raise ValueError(f"obs_multistep_horizon must be positive, got {args.obs_multistep_horizon}.")
    if args.obs_curriculum_epochs <= 0:
        raise ValueError(f"obs_curriculum_epochs must be positive, got {args.obs_curriculum_epochs}.")
    if args.obs_multistep_horizon >= obs_training_steps:
        raise ValueError(
            f"obs_multistep_horizon={args.obs_multistep_horizon} leaves no training samples for observed steps={obs_training_steps}."
        )
    obs_training_horizon = args.obs_multistep_horizon if args.obs_multistep_loss else 1
    obs_ckpt_path = run_path / args.obs_ckpt_name
    obs_ckpt_loaded = obs_ckpt_path.exists()
    train_loader = None
    valid_loader = None
    if not obs_ckpt_loaded:
        train_x, train_y, train_future_inputs = make_observation_prediction_tensors(
            obs_sensors_data,
            idx_train,
            obs_training_steps,
            obs_lag,
            obs_sensor_feature_count,
            obs_input_size,
            forecast_horizon=obs_training_horizon,
        )
        valid_x, valid_y, valid_future_inputs = make_observation_prediction_tensors(
            obs_sensors_data,
            idx_valid if len(idx_valid) > 0 else idx_train,
            obs_training_steps,
            obs_lag,
            obs_sensor_feature_count,
            obs_input_size,
            forecast_horizon=obs_training_horizon,
        )
        train_loader_generator = torch.Generator()
        train_loader_generator.manual_seed(args.obs_seed)
        train_loader = DataLoader(
            TensorDataset(train_x, train_y, train_future_inputs),
            batch_size=args.obs_batch_size,
            shuffle=True,
            generator=train_loader_generator,
        )
        valid_loader = DataLoader(TensorDataset(valid_x, valid_y, valid_future_inputs), batch_size=args.obs_batch_size, shuffle=False)

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

    obs_model = SensorObservationLSTM(
        input_size=obs_input_size,
        sensor_size=obs_sensor_feature_count,
        hidden_size=args.obs_hidden_size,
        num_layers=args.obs_layers,
        dropout=args.obs_dropout,
    ).to(device)
    args._input_size = obs_input_size
    args._sensor_size = obs_sensor_feature_count
    if obs_ckpt_loaded:
        obs_best_val, obs_best_epoch = load_observation_model_checkpoint(obs_model, args, device, obs_ckpt_path)
        print(f"Loaded existing observation LSTM checkpoint from {obs_ckpt_path}")
    else:
        obs_best_val, obs_best_epoch = train_observation_model(obs_model, train_loader, valid_loader, args, device, obs_ckpt_path)
    obs_model.eval()

    cstride_total_params, cstride_trainable_params = get_model_params(cstride_model)
    obs_total_params, obs_trainable_params = get_model_params(obs_model)
    step_sample_rel = [[] for _ in range(eval_horizon)]
    obs_step_sample_rel = [[] for _ in range(eval_horizon)]

    start = time.time()
    with torch.no_grad():
        for entry_idx in eval_indices:
            entry = entries[int(entry_idx)]
            for start_step in start_steps_by_entry[int(entry_idx)]:
                cstride_history = Padding(sensors_data[int(entry_idx), :start_step].unsqueeze(0), cstride_lag)[start_step - 1].unsqueeze(0)
                obs_history = Padding(
                    obs_sensors_data[int(entry_idx), :start_step, :obs_input_size].unsqueeze(0),
                    obs_lag,
                )[start_step - 1].unsqueeze(0)
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
                if include_precipitation or obs_include_precipitation:
                    shared_future_precip = make_future_precip(entry, root, normalizers, start_step, eval_horizon, True)
                    mismatch_seed = make_precip_mismatch_seed(args.eval_seed, seed_indices[int(entry_idx)], start_step)
                    shared_future_precip = impose_precip_mismatch(shared_future_precip, args.precip_mismatch, mismatch_seed)
                else:
                    shared_future_precip = make_future_precip(entry, root, normalizers, start_step, eval_horizon, False)
                future_precip = shared_future_precip if include_precipitation else make_future_precip(entry, root, normalizers, start_step, eval_horizon, False)
                obs_future_precip = shared_future_precip if obs_include_precipitation else make_future_precip(entry, root, normalizers, start_step, eval_horizon, False)

                cstride_history = cstride_history.to(device)
                obs_history = obs_history.to(device)
                batch_coords = batch_coords.to(device)
                batch_target = batch_target.to(device)
                future_precip = future_precip.to(device)
                obs_future_precip = obs_future_precip.to(device)

                pred, pred_observations = autoregressive_forecast_with_observation_predictions(
                    cstride_model,
                    obs_model,
                    cstride_history,
                    obs_history,
                    batch_coords,
                    future_precip,
                    obs_future_precip,
                )

                pred_np = pred.cpu().numpy()
                target_np = batch_target.cpu().numpy()
                pred_denorm = target_normalizer.inverse_transform(pred_np.reshape(-1, output_size)).reshape(pred_np.shape)
                target_denorm = target_normalizer.inverse_transform(target_np.reshape(-1, output_size)).reshape(target_np.shape)

                pred_obs_np = pred_observations.cpu().numpy()
                true_obs_np = (
                    obs_sensors_data[int(entry_idx), start_step : start_step + eval_horizon, :obs_sensor_feature_count]
                    .unsqueeze(0)
                    .cpu()
                    .numpy()
                )
                pred_obs_denorm = observation_normalizer.inverse_transform(
                    pred_obs_np.reshape(-1, len(observable_indices))
                ).reshape(pred_obs_np.shape)
                true_obs_denorm = observation_normalizer.inverse_transform(
                    true_obs_np.reshape(-1, len(observable_indices))
                ).reshape(true_obs_np.shape)
                for step_idx in range(eval_horizon):
                    step_sample_rel[step_idx].append(
                        safe_relative_error(pred_denorm[0, step_idx], target_denorm[0, step_idx])
                    )
                    obs_step_sample_rel[step_idx].append(
                        safe_relative_error(pred_obs_denorm[0, step_idx], true_obs_denorm[0, step_idx])
                    )
    inference_time = time.time() - start

    step_mean = []
    step_std = []
    for values in step_sample_rel:
        mean, std = finite_mean_std(values)
        step_mean.append(mean)
        step_std.append(std)

    obs_step_mean = []
    obs_step_std = []
    for values in obs_step_sample_rel:
        mean, std = finite_mean_std(values)
        obs_step_mean.append(mean)
        obs_step_std.append(std)

    if args.output is not None:
        plot_name = args.output
    else:
        mismatch_slug = precip_mismatch_slug(args.precip_mismatch)
        plot_name = f"reconstruction_obspred_relative_error_{args.split}_h{eval_horizon}{mismatch_slug}.png"
    plot_path = run_path / plot_name
    plot_step_errors(step_mean, step_std, plot_path)

    append_results(
        str(run_path),
        args.split,
        observed_fraction,
        observed_steps,
        eval_horizon,
        include_precipitation,
        obs_include_precipitation,
        args.precip_mismatch,
        terrain_conditioning,
        obs_best_epoch,
        obs_best_val,
        inference_time,
        cstride_total_params,
        cstride_trainable_params,
        obs_total_params,
        obs_trainable_params,
        step_mean,
        step_std,
        obs_step_mean,
        obs_step_std,
        args.obs_seed,
        obs_lag,
        args.obs_lr,
        args.obs_weight_decay,
        args.obs_batch_size,
        args.obs_grad_clip,
        args.obs_multistep_loss,
        args.obs_multistep_horizon,
        args.obs_curriculum,
        args.obs_curriculum_epochs,
        obs_ckpt_loaded,
        args.eval_seed,
        args.num_starts,
        start_steps_by_entry,
        str(plot_path),
    )

    print(f"C-STRIDE model loaded from {ckpt_path}")
    if obs_ckpt_loaded:
        print(f"Observation LSTM checkpoint loaded from {obs_ckpt_path}")
    else:
        print(f"Observation LSTM checkpoint saved to {obs_ckpt_path}")
    print(f"Observation LSTM training seed: {args.obs_seed}")
    print(
        f"Observation LSTM multi-step loss: {args.obs_multistep_loss}, "
        f"horizon={args.obs_multistep_horizon}, curriculum={args.obs_curriculum}"
    )
    print(f"Observed training steps: {observed_steps}/{ntimes} (observed_time_fraction={observed_fraction})")
    print(f"Autoregressive horizon: {eval_horizon}")
    print(f"Precipitation mismatch: {args.precip_mismatch}")
    print(f"Evaluation seed: {args.eval_seed}")
    print(f"Evaluation starts per trajectory: {args.num_starts}; total samples: {sum(len(v) for v in start_steps_by_entry.values())}")
    print(f"Relative error plot saved to {plot_path}")
    for step_idx, (mean, std) in enumerate(zip(step_mean, step_std), start=1):
        print(f"Forecast step {step_idx}: denormalized relative error mean={mean}, std={std}")
    for step_idx, (mean, std) in enumerate(zip(obs_step_mean, obs_step_std), start=1):
        print(f"Forecast step {step_idx}: denormalized observation relative error mean={mean}, std={std}")


if __name__ == "__main__":
    torch.set_default_device("cpu")
    parser = argparse.ArgumentParser(description="Forecast Chicago fields using an auxiliary sensor-observation LSTM and a reconstruction backbone")
    parser.add_argument("--path", required=True, help="Saved C-STRIDE run directory")
    parser.add_argument("--horizon", type=int, default=24, help="Autoregressive rollout horizon")
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--output", default=None, help="Plot filename relative to the run directory")
    parser.add_argument("--obs_ckpt_name", default=None, help="Observation LSTM checkpoint filename; defaults to sensor_observation_lstm.ckpt, or sensor_observation_lstm_precip.ckpt when --obs_include_precipitation_input is set")
    parser.add_argument("--obs_include_precipitation_input", action="store_true", help="Include precipitation as an auxiliary input to the observation LSTM, independent of the backbone config")
    parser.add_argument("--precip_mismatch", type=float, default=0.0, help="Scale of autoregressive future-precipitation mismatch; noise is proportional to per-step true normalized precipitation std and grows linearly over the rollout")
    parser.add_argument("--obs_lag", type=int, default=None, help="Observation LSTM input lag; defaults to basics.lag")
    parser.add_argument("--obs_hidden_size", type=int, default=128)
    parser.add_argument("--obs_layers", type=int, default=2)
    parser.add_argument("--obs_dropout", type=float, default=0.0)
    parser.add_argument("--obs_epochs", type=int, default=1000)
    parser.add_argument("--obs_batch_size", type=int, default=512)
    parser.add_argument("--obs_lr", type=float, default=1e-3)
    parser.add_argument("--obs_weight_decay", type=float, default=1e-4)
    parser.add_argument("--obs_patience", type=int, default=50)
    parser.add_argument("--obs_improvement_threshold", type=float, default=1e-6)
    parser.add_argument("--obs_grad_clip", type=float, default=1.0)
    parser.add_argument("--obs_log_interval", type=int, default=10)
    parser.add_argument("--obs_multistep_loss", action="store_true", default=False, help="Train observation LSTM with autoregressive multi-step loss")
    parser.add_argument("--obs_multistep_horizon", type=int, default=6, help="Maximum loss horizon when --obs_multistep_loss is enabled")
    parser.add_argument("--obs_curriculum", action="store_true", default=False, help="Grow multi-step loss horizon during training")
    parser.add_argument("--obs_curriculum_epochs", type=int, default=100, help="Epochs per horizon increment for --obs_curriculum")
    parser.add_argument("--obs_seed", type=int, default=None, help="Observation LSTM seed; defaults to basics.random_seed")
    parser.add_argument("--eval_seed", type=int, default=None, help="Random seed for evaluation start-step sampling; defaults to basics.random_seed")
    parser.add_argument("--num_starts", type=int, default=10, help="Random valid start steps per evaluation trajectory")
    main(parse_args(parser))
