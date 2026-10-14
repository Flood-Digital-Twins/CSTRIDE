import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from cstride.chicago import (
    load_sensor_map_file,
    add_chicago_input_noise,
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
    make_windows,
    resolve_chicago_observables,
    resolve_chicago_targets,
)
from cstride.chicago_forecast import build_chicago_feature_coords
from cstride.models import CSTRIDE


def load_state_dict_compatible(path, device):
    state_dict = torch.load(path, map_location=device, weights_only=True)
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k[len("_orig_mod.") :]: v for k, v in state_dict.items()}
    return state_dict


def encode_latent(model, history):
    h_0 = torch.zeros((model.hidden_layers, history.size(0), model.hidden_size), device=history.device)
    c_0 = torch.zeros((model.hidden_layers, history.size(0), model.hidden_size), device=history.device)
    _, (latent, _) = model.lstm(history, (h_0, c_0))
    return latent[-1].view(-1, model.hidden_size)


def load_metric_sensor_indices(path, entries, root, target_entry):
    sensor_obj = load_sensor_map_file(path)
    if isinstance(sensor_obj, dict):
        sensor_map = extend_chicago_sensor_map(entries, root, sensor_obj)
        return np.asarray(sensor_map[target_entry["traj_id"]], dtype=np.int64)
    return np.asarray(sensor_obj, dtype=np.int64)


def finite_divide(numerator, denominator):
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def nash_sutcliffe_efficiency(pred, obs):
    denom = np.sum((obs - np.mean(obs)) ** 2)
    return 1.0 - finite_divide(np.sum((pred - obs) ** 2), denom)


def kling_gupta_efficiency(pred, obs):
    pred_std = np.std(pred)
    obs_std = np.std(obs)
    pred_mean = np.mean(pred)
    obs_mean = np.mean(obs)
    if pred_std == 0 or obs_std == 0:
        r = float("nan")
    else:
        r = float(np.corrcoef(pred, obs)[0, 1])
    alpha = finite_divide(pred_std, obs_std)
    beta = finite_divide(pred_mean, obs_mean)
    if not np.isfinite(r) or not np.isfinite(alpha) or not np.isfinite(beta):
        return float("nan")
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def peak_depth_relative_error(pred, obs):
    obs_peak = float(np.max(obs))
    pred_peak = float(np.max(pred))
    return finite_divide(abs(pred_peak - obs_peak), obs_peak)


def classification_metrics(tp, fp, fn):
    precision = finite_divide(tp, tp + fp)
    recall = finite_divide(tp, tp + fn)
    f1 = finite_divide(2 * tp, 2 * tp + fp + fn)
    csi = finite_divide(tp, tp + fp + fn)
    return {
        "csi": csi,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def build_feature_coords_for_entry(entry, root, normalizers, terrain_conditioning):
    coords = np.load(root / entry["coords_file"], mmap_mode="r")
    if terrain_conditioning:
        elevation = np.load(root / entry["elevation_file"], mmap_mode="r")
        manning = np.load(root / entry["manning_file"], mmap_mode="r")
        slope = np.load(root / entry["slope_file"], mmap_mode="r")
    else:
        elevation = None
        manning = None
        slope = None
    return build_chicago_feature_coords(
        coords,
        elevation,
        manning,
        slope,
        normalizers,
        terrain_conditioning=terrain_conditioning,
    )


def evaluate_trajectory(
    model,
    windows,
    feature_coords,
    state,
    target_normalizer,
    target_indices,
    metric_sensor_indices,
    lag,
    device,
    coord_chunk_size,
    inundation_threshold,
):
    depth_output_position = target_indices.index(0)
    nt_valid = windows.shape[0]
    output_size = len(target_indices)
    nstate = feature_coords.shape[0]
    coords_tensor = torch.tensor(feature_coords, dtype=torch.float32)

    usgs_pred = np.zeros((nt_valid, len(metric_sensor_indices)), dtype=np.float32)
    usgs_true = np.asarray(state[lag:, metric_sensor_indices, 0], dtype=np.float32)
    tp = fp = fn = 0

    model.eval()
    with torch.no_grad():
        for time_idx in range(nt_valid):
            history = windows[time_idx : time_idx + 1].to(device)
            latent = encode_latent(model, history)
            true_depth = np.asarray(state[lag + time_idx, :, 0], dtype=np.float32)
            pred_depth = np.empty(nstate, dtype=np.float32)

            for start in range(0, nstate, coord_chunk_size):
                end = min(start + coord_chunk_size, nstate)
                coord_chunk = coords_tensor[start:end].unsqueeze(0).to(device)
                pred_chunk = model.decode(coord_chunk, latent)
                pred_chunk_np = pred_chunk.cpu().numpy().reshape(-1, output_size)
                pred_chunk_denorm = target_normalizer.inverse_transform(pred_chunk_np)
                pred_depth[start:end] = pred_chunk_denorm[:, depth_output_position]

            usgs_pred[time_idx] = pred_depth[metric_sensor_indices]
            pred_inundated = pred_depth >= inundation_threshold
            true_inundated = true_depth >= inundation_threshold
            tp += int(np.count_nonzero(pred_inundated & true_inundated))
            fp += int(np.count_nonzero(pred_inundated & ~true_inundated))
            fn += int(np.count_nonzero(~pred_inundated & true_inundated))

    sensor_metrics = []
    for sensor_pos, sensor_idx in enumerate(metric_sensor_indices):
        pred_series = usgs_pred[:, sensor_pos]
        true_series = usgs_true[:, sensor_pos]
        sensor_metrics.append(
            {
                "sensor_position": sensor_pos,
                "grid_index": int(sensor_idx),
                "nse": nash_sutcliffe_efficiency(pred_series, true_series),
                "kge": kling_gupta_efficiency(pred_series, true_series),
                "peak_depth_relative_error": peak_depth_relative_error(pred_series, true_series),
                "true_peak_depth": float(np.max(true_series)),
                "predicted_peak_depth": float(np.max(pred_series)),
            }
        )

    return sensor_metrics, classification_metrics(tp, fp, fn), {"tp": tp, "fp": fp, "fn": fn}


def write_results(output_path, run_path, trajectory_id, lag, threshold, sensor_path, sensor_metrics, inundation_metrics, counts):
    with open(output_path, "w") as f:
        f.write("Chicago Additional Evaluation\n")
        f.write("=============================\n")
        f.write(f"Run path: {run_path}\n")
        f.write(f"Trajectory id: {trajectory_id}\n")
        f.write(f"Predicted time range: lag..ntimes-1, lag={lag}\n")
        f.write(f"USGS sensor path: {sensor_path}\n")
        f.write(f"Inundation threshold: {threshold} m\n\n")
        f.write("USGS Sensor Time-Series Metrics\n")
        f.write("--------------------------------\n")
        for item in sensor_metrics:
            f.write(
                f"Sensor {item['sensor_position']} (grid_index={item['grid_index']}): "
                f"NSE={item['nse']}, KGE={item['kge']}, "
                f"peak_depth_relative_error={item['peak_depth_relative_error']}, "
                f"true_peak_depth={item['true_peak_depth']}, "
                f"predicted_peak_depth={item['predicted_peak_depth']}\n"
            )
        f.write("\nInundation Metrics Aggregated Over Time And Space\n")
        f.write("-------------------------------------------------\n")
        f.write(f"TP={counts['tp']}, FP={counts['fp']}, FN={counts['fn']}\n")
        f.write(f"CSI={inundation_metrics['csi']}\n")
        f.write(f"F1={inundation_metrics['f1']}\n")
        f.write(f"Precision={inundation_metrics['precision']}\n")
        f.write(f"Recall={inundation_metrics['recall']}\n")


def main(args):
    run_path = Path(args.path)
    with open(run_path / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    if config["basics"].get("scaler", "global") != "global":
        raise ValueError("Chicago additional evaluation currently supports only global scaling.")

    root, manifest, entries = load_chicago_manifest(config["basics"].get("data_root", "data/Chicago"))
    target_matches = [idx for idx, entry in enumerate(entries) if int(entry["traj_id"]) == int(args.trajectory_id)]
    if not target_matches:
        raise ValueError(f"Trajectory id {args.trajectory_id} was not found in the Chicago manifest.")
    entry_pos = int(target_matches[0])
    entry = entries[entry_pos]

    dataset_output_size = get_chicago_output_size(root, entries[0])
    target_indices = resolve_chicago_targets(config, dataset_output_size)
    if 0 not in target_indices:
        raise ValueError("Additional evaluation requires depth channel 0 to be included in target_variables.")
    observable_indices = resolve_chicago_observables(config, dataset_output_size)
    include_precipitation = chicago_include_precipitation_input(config)
    terrain_conditioning = chicago_use_terrain_conditioning(config)
    coord_size = 5 if terrain_conditioning else 2
    lag = int(config["basics"]["lag"])

    idx_train, _, _ = chicago_split_indices(entries)
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
    sensor_feature_count = int(config["basics"].get("num_sensors", 0)) * len(observable_indices)
    if config["basics"].get("noise", 0.0) > 0.0:
        sensors_data = add_chicago_input_noise(
            sensors_data,
            config["basics"]["noise"],
            sensor_feature_count=sensor_feature_count,
        )
    windows = make_windows(sensors_data[entry_pos], lag)

    device = torch.device(config["basics"]["device"] if torch.cuda.is_available() else "cpu")
    model = CSTRIDE(
        input_size=sensors_data.shape[-1],
        hidden_size=config["model"]["hidden_size"],
        hidden_layers=config["model"]["hidden_layers"],
        fmmnn_depth=config["model"]["fmmnn_depth"],
        fmmnn_width=config["model"]["fmmnn_width"],
        fmmnn_rank=config["model"]["fmmnn_rank"],
        coord_size=coord_size,
        output_size=len(target_indices),
        num_fourier=config["model"]["num_fourier"],
        device=device,
    ).to(device)

    ckpt_path = run_path / "best_model.ckpt"
    if not ckpt_path.exists():
        ckpt_path = run_path / "final_model.ckpt"
    model.load_state_dict(load_state_dict_compatible(ckpt_path, device))

    usgs_sensor_path = args.usgs_sensor_path or str(ROOT / "data/gauges/USGS_6sensors.json")
    metric_sensor_indices = load_metric_sensor_indices(usgs_sensor_path, entries, root, entry)
    state = get_chicago_state_array(root, entry)
    feature_coords = build_feature_coords_for_entry(entry, root, normalizers, terrain_conditioning)

    sensor_metrics, inundation_metrics, counts = evaluate_trajectory(
        model=model,
        windows=windows,
        feature_coords=feature_coords,
        state=state,
        target_normalizer=target_normalizer,
        target_indices=target_indices,
        metric_sensor_indices=metric_sensor_indices,
        lag=lag,
        device=device,
        coord_chunk_size=args.coord_chunk_size,
        inundation_threshold=args.inundation_threshold,
    )

    output_path = run_path / args.output
    write_results(
        output_path,
        str(run_path),
        int(entry["traj_id"]),
        lag,
        args.inundation_threshold,
        usgs_sensor_path,
        sensor_metrics,
        inundation_metrics,
        counts,
    )

    print(f"Additional evaluation written to {output_path}")
    for item in sensor_metrics:
        print(
            f"Sensor {item['sensor_position']} NSE={item['nse']}, KGE={item['kge']}, "
            f"peak_depth_relative_error={item['peak_depth_relative_error']}"
        )
    print(
        f"CSI={inundation_metrics['csi']}, F1={inundation_metrics['f1']}, "
        f"Precision={inundation_metrics['precision']}, Recall={inundation_metrics['recall']}"
    )


if __name__ == "__main__":
    torch.set_default_device("cpu")
    parser = argparse.ArgumentParser(description="Compute additional Chicago evaluation metrics for a saved C-STRIDE run")
    parser.add_argument("--path", required=True, help="Saved C-STRIDE run directory")
    parser.add_argument("--trajectory_id", type=int, default=120)
    parser.add_argument("--usgs_sensor_path", default=None, help="USGS sensor index file; defaults to data/Chicago/USGS_6sensors.pt")
    parser.add_argument("--output", default="additional_evaluation_traj120.txt")
    parser.add_argument("--inundation_threshold", type=float, default=0.5)
    parser.add_argument("--coord_chunk_size", type=int, default=200000)
    main(parser.parse_args())
