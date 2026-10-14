from .runtime import ROOT, load_config, parse_args, load_state_dict
import argparse
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .chicago import (
    build_sensor_sequences,
    chicago_include_precipitation_input,
    chicago_split_indices,
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
from .chicago_forecast import (
    ChicagoForecastWindowDataset,
    build_chicago_feature_coords,
    make_forecast_histories,
    make_future_exogenous,
)
from .evaluate import get_model_params
from .forecast import autoregressive_forecast
from .models import CSTRIDEForecast
from .processdata import Padding, mre, num2p


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


def split_indices_for_name(split_name, entries, split_file=None):
    idx_train, idx_valid, idx_test = chicago_split_indices(entries, split_file)
    if split_name == "train":
        return idx_train
    if split_name == "valid":
        return idx_valid
    if split_name == "test":
        return idx_test
    raise ValueError(f"Unsupported split: {split_name}")


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


def build_sampled_rollout_tensors(
    entry,
    root,
    sensor_indices,
    start_step,
    eval_horizon,
    normalizers,
    target_indices,
    target_normalizer,
    terrain_conditioning,
    include_precipitation,
):
    if hasattr(sensor_indices, "detach"):
        sensor_indices = sensor_indices.detach().cpu().numpy()
    sensor_indices = np.asarray(sensor_indices, dtype=np.int64)
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

    sensor_feature_coords = build_chicago_feature_coords(
        coords[sensor_indices],
        elevation[sensor_indices] if terrain_conditioning else None,
        manning[sensor_indices] if terrain_conditioning else None,
        slope[sensor_indices] if terrain_conditioning else None,
        normalizers,
        terrain_conditioning=terrain_conditioning,
    )
    future_sensor_coords = np.repeat(sensor_feature_coords[None, None, :, :], eval_horizon, axis=1)

    state = get_chicago_state_array(root, entry)
    target = state[start_step : start_step + eval_horizon, :, :][:, :, target_indices]
    target = target_normalizer.transform(target).astype(np.float32)[None]

    if include_precipitation:
        precip = np.load(root / entry["precip_file"], mmap_mode="r")
        precip = normalizers["precip"].transform(precip)
        future_forcing = precip[start_step : start_step + eval_horizon][None].astype(np.float32)
    else:
        future_forcing = np.empty((1, eval_horizon, 0), dtype=np.float32)

    return (
        torch.tensor(future_coords, dtype=torch.float32),
        torch.tensor(target, dtype=torch.float32),
        torch.tensor(future_sensor_coords, dtype=torch.float32),
        torch.tensor(future_forcing, dtype=torch.float32),
    )

def append_results(path, total_params, trainable_params, inference_time, rel_error_norm, mean_error_norm, step_norm, step_depth):
    with open(os.path.join(path, "results.txt"), "a") as f:
        f.write("\nChicago Forecast Evaluation\n")
        f.write("===========================\n")
        f.write(f"Total parameters in the model: {total_params}\n")
        f.write(f"Trainable parameters in the model: {trainable_params}\n")
        f.write(f"Inference time: {inference_time:.2f} seconds\n")
        f.write(f"Relative prediction error on U (Normalized): {rel_error_norm}\n")
        f.write(f"Mean relative prediction error on U (Normalized): {mean_error_norm}\n")
        for step_idx, rel_error in enumerate(step_norm, start=1):
            f.write(f"Relative prediction error on U (Normalized) at forecast step {step_idx}: {rel_error}\n")
        for step_idx, metrics in enumerate(step_depth, start=1):
            f.write(f"MRE on depth U0 (denormalized) at forecast step {step_idx}: {metrics['mre']}\n")
            f.write(f"Relative prediction error on depth U0 at forecast step {step_idx}: {metrics['rel_error']}\n")



def append_sampled_autoregressive_results(
    path,
    split_name,
    eval_horizon,
    eval_seed,
    num_starts,
    start_steps_by_entry,
    inference_time,
    step_mean,
    step_std,
    plot_path,
):
    with open(os.path.join(path, "results.txt"), "a") as f:
        f.write("\nChicago Forecast Sampled Autoregressive Evaluation\n")
        f.write("==================================================\n")
        f.write(f"Split: {split_name}\n")
        f.write(f"Autoregressive horizon: {eval_horizon}\n")
        f.write(f"Evaluation seed: {eval_seed}\n")
        f.write(f"Random valid starts per trajectory: {num_starts}\n")
        f.write("Comparable setting: history uses true inputs strictly before each start step.\n")
        f.write("Comparable setting: targets begin at the sampled start step, matching obspred/latentpred.\n")
        f.write("Comparable setting: future sensor observations are model-predicted autoregressively, not read from data.\n")
        for entry_idx, starts in start_steps_by_entry.items():
            f.write(f"Evaluation start steps for trajectory index {entry_idx}: {starts.tolist()}\n")
        f.write(f"Inference time: {inference_time:.2f} seconds\n")
        f.write(f"Relative error plot: {plot_path}\n")
        for step_idx, (mean, std) in enumerate(zip(step_mean, step_std), start=1):
            f.write(
                f"Forecast step {step_idx} denormalized relative error with autoregressive prediction: "
                f"mean={mean} ({num2p_safe(mean)}), std={std}\n"
            )

def main(args):
    path = args.path
    with open(os.path.join(path, "config.yaml"), "r") as f:
        config = load_config(f)
    if config["basics"].get("scaler", "global") != "global":
        raise ValueError("Chicago forecast evaluation currently supports only global scaling.")

    root, manifest, entries = load_chicago_manifest(config["basics"].get("data_root", "data/Chicago"))
    ntimes = int(manifest["time_steps"])
    horizon = int(config["forecast"]["horizon"])
    sampled_horizon = int(args.ar_horizon)
    if sampled_horizon <= 0:
        raise ValueError(f"Autoregressive evaluation horizon must be positive, got {sampled_horizon}.")
    eval_seed = int(args.ar_seed) if args.ar_seed is not None else int(config["basics"].get("random_seed", 0))
    lag = int(config["basics"]["lag"])
    dataset_output_size = get_chicago_output_size(root, entries[0])
    observable_indices = resolve_chicago_observables(config, dataset_output_size)
    target_indices = resolve_chicago_targets(config, dataset_output_size)
    observable_positions = observable_output_positions(observable_indices, target_indices)
    if 0 not in target_indices:
        raise ValueError("Chicago forecast evaluation currently reports depth metrics, but target_variables excludes channel 0.")
    depth_target_pos = target_indices.index(0)
    output_size = len(target_indices)
    include_precipitation = chicago_include_precipitation_input(config)
    terrain_conditioning = chicago_use_terrain_conditioning(config)
    coord_size = 5 if terrain_conditioning else 2

    idx_train, _, idx_test = chicago_split_indices(entries, config["basics"].get("split_file"))
    normalizers = fit_chicago_normalizers(entries, idx_train, root)
    target_normalizer = get_chicago_target_normalizer(normalizers, target_indices)
    sensor_map = load_chicago_sensors(path, config)
    if sensor_map is None:
        raise FileNotFoundError(f"No saved Chicago sensor mapping found in {path}.")

    sensor_sequences = build_sensor_sequences(
        entries,
        root,
        sensor_map,
        normalizers,
        observable_indices,
        include_precipitation=include_precipitation,
    )
    sensors_data = torch.stack(sensor_sequences, dim=0)
    test_entries = [entries[i] for i in idx_test]
    test_sensors = sensors_data[idx_test]
    test_windows = make_forecast_histories(test_sensors, lag, horizon)
    test_future_forcing = make_future_exogenous(
        test_entries,
        root,
        normalizers,
        horizon,
        include_precipitation=include_precipitation,
    )
    test_dataset = ChicagoForecastWindowDataset(
        test_entries,
        root,
        test_windows,
        test_future_forcing,
        ntimes - horizon,
        horizon,
        normalizers,
        target_indices,
        sensor_map,
        terrain_conditioning=terrain_conditioning,
        num_points=None,
        spatial_probs=None,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device(config["basics"]["device"] if torch.cuda.is_available() else "cpu")
    model = CSTRIDEForecast(
        input_size=test_dataset.x_data.shape[-1],
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
    ckpt_path = os.path.join(path, "best_model.ckpt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(path, "final_model.ckpt")
    model.load_state_dict(load_state_dict_compatible(ckpt_path, device))
    model.eval()

    total_params, trainable_params = get_model_params(model)
    sum_gt = 0.0
    sum_err = 0.0
    step_gt = np.zeros(horizon, dtype=np.float64)
    step_err = np.zeros(horizon, dtype=np.float64)
    sample_rel_norm = []
    depth_step_sample_rel = [[] for _ in range(horizon)]
    depth_step_sum_gt = np.zeros(horizon, dtype=np.float64)
    depth_step_sum_err = np.zeros(horizon, dtype=np.float64)
    start = time.time()
    with torch.no_grad():
        for batch_x, batch_coords, batch_target, batch_sensor_coords, batch_future_forcing in test_loader:
            batch_x = batch_x.to(device)
            batch_coords = batch_coords.to(device)
            batch_target = batch_target.to(device)
            batch_sensor_coords = batch_sensor_coords.to(device)
            batch_future_forcing = batch_future_forcing.to(device)
            pred = autoregressive_forecast(
                model,
                batch_x,
                batch_coords,
                batch_sensor_coords,
                observables=observable_positions,
                future_exogenous=batch_future_forcing,
            )
            sum_gt += torch.sum(batch_target**2).item()
            sum_err += torch.sum((pred - batch_target) ** 2).item()
            step_gt += torch.sum(batch_target**2, dim=(0, 2, 3)).cpu().numpy()
            step_err += torch.sum((pred - batch_target) ** 2, dim=(0, 2, 3)).cpu().numpy()

            pred_np = pred.cpu().numpy()
            target_np = batch_target.cpu().numpy()
            for batch_idx in range(pred_np.shape[0]):
                sample_gt = np.sum(target_np[batch_idx] ** 2)
                sample_err = np.sum((pred_np[batch_idx] - target_np[batch_idx]) ** 2)
                if sample_gt > 0:
                    sample_rel_norm.append(np.sqrt(sample_err / sample_gt))

            pred_denorm = target_normalizer.inverse_transform(pred_np.reshape(-1, output_size)).reshape(pred_np.shape)
            target_denorm = target_normalizer.inverse_transform(target_np.reshape(-1, output_size)).reshape(target_np.shape)
            for step_idx in range(horizon):
                for batch_idx in range(pred_denorm.shape[0]):
                    step_target = target_denorm[batch_idx, step_idx, :, depth_target_pos]
                    step_pred = pred_denorm[batch_idx, step_idx, :, depth_target_pos]
                    gt = np.sum(step_target**2)
                    err = np.sum((step_pred - step_target) ** 2)
                    depth_step_sum_gt[step_idx] += gt
                    depth_step_sum_err[step_idx] += err
                    if gt > 0:
                        depth_step_sample_rel[step_idx].append(np.sqrt(err / gt))
    inference_time = time.time() - start

    rel_error_norm = float(np.sqrt(sum_err / sum_gt))
    step_norm = np.sqrt(step_err / step_gt)
    mean_error_norm = num2p(float(np.mean(sample_rel_norm))) if sample_rel_norm else "nan%"

    step_depth = []
    for step_idx in range(horizon):
        denom = depth_step_sum_gt[step_idx]
        rel_error = float(np.sqrt(depth_step_sum_err[step_idx] / denom)) if denom > 0 else float("nan")
        step_depth.append(
            {
                "mre": num2p(float(np.mean(depth_step_sample_rel[step_idx]))) if depth_step_sample_rel[step_idx] else "nan%",
                "rel_error": rel_error,
            }
        )

    print(f"Model loaded from {ckpt_path}")
    print(f"Total parameters in the model: {total_params}")
    print(f"Trainable parameters in the model: {trainable_params}")
    print(f"Inference time: {inference_time:.2f} seconds")
    print(f"Relative prediction error on U (Normalized): {rel_error_norm}")
    print(f"Mean relative prediction error on U (Normalized): {mean_error_norm}")
    for step_idx, rel_error in enumerate(step_norm, start=1):
        print(f"Relative prediction error on U (Normalized) at forecast step {step_idx}: {rel_error}")
    for step_idx, metrics in enumerate(step_depth, start=1):
        print(f"MRE on depth U0 (denormalized) at forecast step {step_idx}: {metrics['mre']}")
        print(f"Relative prediction error on depth U0 at forecast step {step_idx}: {metrics['rel_error']}")


    append_results(path, total_params, trainable_params, inference_time, rel_error_norm, mean_error_norm, step_norm, step_depth)

    sampled_indices = split_indices_for_name(args.ar_split, entries, config["basics"].get("split_file"))
    start_steps_by_entry = sample_start_steps(
        sampled_indices,
        ntimes,
        sampled_horizon,
        max(lag, 1),
        int(args.ar_num_starts),
        eval_seed,
    )
    sampled_step_rel = [[] for _ in range(sampled_horizon)]
    sampled_start = time.time()
    with torch.no_grad():
        for entry_idx in sampled_indices:
            entry_idx = int(entry_idx)
            entry = entries[entry_idx]
            sensor_indices = sensor_map[entry["traj_id"]]
            for start_step in start_steps_by_entry[entry_idx]:
                batch_x = Padding(sensors_data[entry_idx, :start_step].unsqueeze(0), lag)[start_step - 1].unsqueeze(0)
                batch_coords, batch_target, batch_sensor_coords, batch_future_forcing = build_sampled_rollout_tensors(
                    entry,
                    root,
                    sensor_indices,
                    int(start_step),
                    sampled_horizon,
                    normalizers,
                    target_indices,
                    target_normalizer,
                    terrain_conditioning,
                    include_precipitation,
                )

                batch_x = batch_x.to(device)
                batch_coords = batch_coords.to(device)
                batch_target = batch_target.to(device)
                batch_sensor_coords = batch_sensor_coords.to(device)
                batch_future_forcing = batch_future_forcing.to(device)
                pred = autoregressive_forecast(
                    model,
                    batch_x,
                    batch_coords,
                    batch_sensor_coords,
                    observables=observable_positions,
                    future_exogenous=batch_future_forcing,
                )

                pred_np = pred.cpu().numpy()
                target_np = batch_target.cpu().numpy()
                pred_denorm = target_normalizer.inverse_transform(pred_np.reshape(-1, output_size)).reshape(pred_np.shape)
                target_denorm = target_normalizer.inverse_transform(target_np.reshape(-1, output_size)).reshape(target_np.shape)
                for step_idx in range(sampled_horizon):
                    sampled_step_rel[step_idx].append(
                        safe_relative_error(pred_denorm[0, step_idx], target_denorm[0, step_idx])
                    )
    sampled_inference_time = time.time() - sampled_start

    sampled_step_mean = []
    sampled_step_std = []
    for values in sampled_step_rel:
        mean, std = finite_mean_std(values)
        sampled_step_mean.append(mean)
        sampled_step_std.append(std)

    plot_name = args.ar_output or f"forecast_autoregressive_relative_error_{args.ar_split}_h{sampled_horizon}.png"
    plot_path = os.path.join(path, plot_name)
    plot_step_errors(sampled_step_mean, sampled_step_std, plot_path)
    append_sampled_autoregressive_results(
        path,
        args.ar_split,
        sampled_horizon,
        eval_seed,
        args.ar_num_starts,
        start_steps_by_entry,
        sampled_inference_time,
        sampled_step_mean,
        sampled_step_std,
        plot_path,
    )

    print(f"Sampled autoregressive evaluation split: {args.ar_split}")
    print(f"Sampled autoregressive horizon: {sampled_horizon}")
    print(f"Evaluation seed: {eval_seed}")
    print(f"Evaluation starts per trajectory: {args.ar_num_starts}; total samples: {sum(len(v) for v in start_steps_by_entry.values())}")
    print("Comparable setting: true history before start, targets from start, future sensors predicted autoregressively")
    print(f"Sampled autoregressive inference time: {sampled_inference_time:.2f} seconds")
    print(f"Sampled autoregressive relative error plot saved to {plot_path}")
    for step_idx, (mean, std) in enumerate(zip(sampled_step_mean, sampled_step_std), start=1):
        print(f"Sampled autoregressive forecast step {step_idx}: denormalized relative error mean={mean}, std={std}")


if __name__ == "__main__":
    torch.set_default_device("cpu")
    parser = argparse.ArgumentParser(description="Evaluate C-STRIDE autoregressive model on the Chicago dataset")
    parser.add_argument("--path", required=True)
    parser.add_argument("--ar_horizon", type=int, default=24, help="Sampled autoregressive rollout horizon")
    parser.add_argument("--ar_split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--ar_num_starts", type=int, default=10, help="Random valid starts per trajectory")
    parser.add_argument("--ar_seed", type=int, default=None, help="Random seed for sampled autoregressive starts; defaults to basics.random_seed")
    parser.add_argument("--ar_output", default=None, help="Sampled autoregressive plot filename relative to the run directory")
    args = parse_args(parser)
    main(args)
