import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import argparse
import zipfile
from pathlib import Path

import numpy as np
from numpy.lib import format as npy_format
import yaml

from score_additional_metrics import (
    classification_metrics,
    kling_gupta_efficiency,
    load_metric_sensor_indices,
    nash_sutcliffe_efficiency,
    peak_depth_relative_error,
)
from cstride.chicago import (
    chicago_split_indices,
    get_chicago_output_size,
    get_chicago_state_array,
    load_chicago_manifest,
    resolve_chicago_targets,
)
from cstride.processdata import num2p


PREDICTION_PATTERN = "predictions_traj{traj_id}.npz"


def finite_relative_error(error_sum, gt_sum):
    if gt_sum <= 0:
        return float("nan")
    return float(np.sqrt(error_sum / gt_sum))


def open_prediction_member(npz_path, member_name):
    archive = zipfile.ZipFile(npz_path)
    member = archive.open(member_name)
    version = npy_format.read_magic(member)
    shape, fortran_order, dtype = npy_format._read_array_header(member, version)
    if fortran_order:
        member.close()
        archive.close()
        raise ValueError(f"{npz_path}:{member_name} is Fortran-order; expected C-order predictions.")
    return archive, member, tuple(shape), np.dtype(dtype)


def read_prediction_time_slice(member, height, width, channels, dtype):
    nbytes = height * width * channels * dtype.itemsize
    data = member.read(nbytes)
    if len(data) != nbytes:
        raise EOFError(f"Expected {nbytes} prediction bytes, got {len(data)}")
    return np.frombuffer(data, dtype=dtype).reshape(height, width, channels)


def load_dense_truth_time_series(truth_dir, traj_id):
    if truth_dir is None:
        return None
    truth_path = Path(truth_dir) / f"truth_traj{traj_id}.npy"
    if not truth_path.exists():
        raise FileNotFoundError(
            f"Missing dense truth file for trajectory {traj_id}: {truth_path}. "
            "Full-grid baseline predictions should be evaluated against matching full-grid truth; "
            "set --allow_sparse_truth only if you intentionally want sparse cell-center sampling."
        )
    truth = np.load(truth_path, mmap_mode="r")
    if truth.ndim != 4:
        raise ValueError(f"Expected dense truth shape (time, height, width, channels), got {truth.shape} from {truth_path}")
    return truth_path, truth


def dense_indices_from_coords(coords, height, width, extent=None):
    coords = np.asarray(coords, dtype=np.float32)
    aspect = height / float(width)
    cols = np.rint((coords[:, 0] + 1.0) * width / 2.0).astype(np.int64)
    rows = np.rint((coords[:, 1] / aspect + 1.0) * height / 2.0).astype(np.int64)
    valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    if not np.all(valid):
        raise ValueError(f"Dense-grid mapping produced {int(np.count_nonzero(~valid))} out-of-bounds cells.")
    return rows, cols


def resolve_dense_flow_path(source_dense_roots, traj_id):
    for root in source_dense_roots or []:
        root = Path(root)
        direct_path = root / "flow_variables.npy"
        if direct_path.exists() and (str(traj_id) in root.name or int(traj_id) == 120):
            return direct_path
        path = root / f"event_{traj_id}_filling_hours_0_filling_intensity_mm_hr_0" / "flow_variables.npy"
        if path.exists():
            return path
    return None


def dense_indices_from_flow_variables(flow_path, height, width, expected_count, burn_in, depth_threshold):
    flow = np.load(flow_path, mmap_mode="r")
    if flow.ndim != 4:
        raise ValueError(f"Unexpected flow_variables shape {flow.shape} from {flow_path}")
    if flow.shape[1] == 3:
        if flow.shape[2:4] != (height, width):
            raise ValueError(f"flow_variables grid {flow.shape[2:4]} does not match prediction grid {(height, width)}")
        depth = flow[burn_in:, 0, :, :]
    elif flow.shape[-1] == 3:
        if flow.shape[1:3] != (height, width):
            raise ValueError(f"flow_variables grid {flow.shape[1:3]} does not match prediction grid {(height, width)}")
        depth = flow[burn_in:, :, :, 0]
    else:
        raise ValueError(f"Unexpected flow_variables channel layout {flow.shape} from {flow_path}")

    valid = np.ones((height, width), dtype=bool)
    max_depth = np.full((height, width), -np.inf, dtype=np.float32)
    for time_idx in range(depth.shape[0]):
        depth_t = np.asarray(depth[time_idx], dtype=np.float32)
        valid &= ~np.isnan(depth_t)
        max_depth = np.maximum(max_depth, np.nan_to_num(depth_t, nan=-np.inf))
    valid &= max_depth >= float(depth_threshold)

    flat_indices = np.flatnonzero(valid.reshape(-1))
    if flat_indices.size != expected_count:
        raise ValueError(
            f"Dense flow valid-cell count {flat_indices.size} from {flow_path} does not match "
            f"sparse truth count {expected_count}."
        )
    return flat_indices // width, flat_indices % width


def init_metric_accumulators(output_size):
    return {
        "total_gt_denorm": np.zeros(output_size, dtype=np.float64),
        "total_err_denorm": np.zeros(output_size, dtype=np.float64),
        "sample_rel_denorm": [[] for _ in range(output_size)],
        "sample_rmse_denorm": [[] for _ in range(output_size)],
        "peak_rel_denorm": [],
    }


def update_reconstruction_metrics(metrics, pred_denorm, target_denorm, time_idx, peak_range):
    output_size = pred_denorm.shape[-1]
    for channel_idx in range(output_size):
        sq_err = (pred_denorm[:, channel_idx] - target_denorm[:, channel_idx]) ** 2
        err = np.sum(sq_err, dtype=np.float64)
        gt = np.sum(target_denorm[:, channel_idx] ** 2, dtype=np.float64)
        metrics["total_err_denorm"][channel_idx] += err
        metrics["total_gt_denorm"][channel_idx] += gt
        metrics["sample_rmse_denorm"][channel_idx].append(float(np.sqrt(np.mean(sq_err))))
        if gt > 0:
            metrics["sample_rel_denorm"][channel_idx].append(float(np.sqrt(err / gt)))

    if peak_range is not None:
        peak_start, peak_end = peak_range
        peak_gt = np.sum(target_denorm[:, 0] ** 2, dtype=np.float64)
        peak_err = np.sum((pred_denorm[:, 0] - target_denorm[:, 0]) ** 2, dtype=np.float64)
        if peak_start <= time_idx < peak_end and peak_gt > 0:
            metrics["peak_rel_denorm"].append(float(np.sqrt(peak_err / peak_gt)))


def summarize_reconstruction_metrics(metrics):
    return {
        "mean_errors_denorm": [
            num2p(float(np.mean(channel_errors))) if channel_errors else "nan%"
            for channel_errors in metrics["sample_rel_denorm"]
        ],
        "mean_rmse_denorm": [
            float(np.mean(channel_rmse)) if channel_rmse else float("nan")
            for channel_rmse in metrics["sample_rmse_denorm"]
        ],
        "rel_errors_denorm": [
            finite_relative_error(err, gt)
            for err, gt in zip(metrics["total_err_denorm"], metrics["total_gt_denorm"])
        ],
        "mre_denorm": num2p(float(np.mean(metrics["sample_rel_denorm"][0])))
        if metrics["sample_rel_denorm"] and metrics["sample_rel_denorm"][0]
        else "nan%",
        "mean_rmse_u0_denorm": float(np.mean(metrics["sample_rmse_denorm"][0]))
        if metrics["sample_rmse_denorm"] and metrics["sample_rmse_denorm"][0]
        else float("nan"),
        "rel_error_denorm": finite_relative_error(metrics["total_err_denorm"][0], metrics["total_gt_denorm"][0])
        if len(metrics["total_gt_denorm"]) > 0
        else float("nan"),
        "peak_mre_denorm": num2p(float(np.mean(metrics["peak_rel_denorm"]))) if metrics["peak_rel_denorm"] else "nan%",
    }


def merge_reconstruction_metrics(metric_accumulators):
    if not metric_accumulators:
        raise ValueError("No trajectory metrics were accumulated.")
    output_size = len(metric_accumulators[0]["total_gt_denorm"])
    merged = init_metric_accumulators(output_size)
    for metrics in metric_accumulators:
        merged["total_gt_denorm"] += metrics["total_gt_denorm"]
        merged["total_err_denorm"] += metrics["total_err_denorm"]
        merged["peak_rel_denorm"].extend(metrics["peak_rel_denorm"])
        for channel_idx in range(output_size):
            merged["sample_rel_denorm"][channel_idx].extend(metrics["sample_rel_denorm"][channel_idx])
            merged["sample_rmse_denorm"][channel_idx].extend(metrics["sample_rmse_denorm"][channel_idx])
    return merged


def evaluate_baseline_trajectory(
    prediction_path,
    entry,
    root,
    target_indices,
    extent,
    truth_dir,
    allow_sparse_truth,
    member_name,
    lag,
    inundation_threshold,
    metric_sensor_indices,
    source_dense_roots=None,
    valid_mask_mode="auto",
    burn_in=1,
    depth_threshold=0.1,
):
    archive, member, pred_shape, dtype = open_prediction_member(prediction_path, member_name)
    try:
        if len(pred_shape) != 4:
            raise ValueError(f"Expected baseline prediction shape (time, height, width, channels), got {pred_shape}")
        ntimes, height, width, pred_channels = pred_shape
        if max(target_indices) >= pred_channels:
            raise ValueError(f"target_indices={target_indices} exceed prediction channels={pred_channels}")

        dense_truth = load_dense_truth_time_series(truth_dir, int(entry["traj_id"])) if truth_dir and not allow_sparse_truth else None
        dense_sensor_indices = None
        if dense_truth is not None:
            truth_path, truth = dense_truth
            if truth.shape[:3] != (ntimes, height, width):
                raise ValueError(
                    f"Dense truth shape {truth.shape} from {truth_path} does not match "
                    f"prediction shape {pred_shape}."
                )
            if max(target_indices) >= truth.shape[-1]:
                raise ValueError(f"target_indices={target_indices} exceed dense truth channels={truth.shape[-1]}")
            rows = cols = state = None
            if metric_sensor_indices is not None:
                coords = np.load(root / entry["coords_file"], mmap_mode="r")
                sensor_coords = coords[np.asarray(metric_sensor_indices, dtype=np.int64)]
                sensor_rows, sensor_cols = dense_indices_from_coords(sensor_coords, height, width, extent)
                dense_sensor_indices = sensor_rows * width + sensor_cols
        else:
            if not allow_sparse_truth:
                raise FileNotFoundError(
                    f"No dense truth directory was provided for trajectory {entry['traj_id']}. "
                    "Pass --truth_dir with truth_traj*.npy files, or pass --allow_sparse_truth "
                    "to use sparse cell-center sampling."
                )
            truth_path = None
            coords = np.load(root / entry["coords_file"], mmap_mode="r")
            state = get_chicago_state_array(root, entry)
            if state.shape[0] != ntimes:
                raise ValueError(
                    f"Baseline time count {ntimes} does not match truth time count {state.shape[0]} "
                    f"for trajectory {entry['traj_id']}"
                )
            dense_flow_path = resolve_dense_flow_path(source_dense_roots, int(entry["traj_id"]))
            if valid_mask_mode in ("auto", "dense_flow") and dense_flow_path is not None:
                try:
                    rows, cols = dense_indices_from_flow_variables(
                        dense_flow_path,
                        height,
                        width,
                        expected_count=state.shape[1],
                        burn_in=burn_in,
                        depth_threshold=depth_threshold,
                    )
                    truth_path = f"{root / entry['state_file']} (valid mask from {dense_flow_path})" if entry.get("state_file") else f"{root / entry['depth_file']} (valid mask from {dense_flow_path})"
                except Exception as exc:
                    if valid_mask_mode == "dense_flow":
                        raise
                    print(
                        f"Falling back to coordinate mapping for trajectory {entry['traj_id']}: "
                        f"could not use {dense_flow_path} ({exc})"
                    )
                    rows, cols = dense_indices_from_coords(coords, height, width, extent)
            elif valid_mask_mode == "dense_flow":
                raise FileNotFoundError(f"No dense flow_variables.npy found for trajectory {entry['traj_id']} in {source_dense_roots}")
            else:
                rows, cols = dense_indices_from_coords(coords, height, width, extent)

        metrics = init_metric_accumulators(len(target_indices))
        depth_output_position = target_indices.index(0) if 0 in target_indices else None
        peak_range = (48 - lag, 84 - lag) if depth_output_position == 0 else None
        usgs_pred = []
        usgs_true = []
        tp = fp = fn = 0

        for time_idx in range(ntimes):
            pred_grid = read_prediction_time_slice(member, height, width, pred_channels, dtype)
            if time_idx < lag:
                continue

            if dense_truth is not None:
                pred_denorm = np.asarray(pred_grid[:, :, target_indices].reshape(-1, len(target_indices)), dtype=np.float32)
                target_denorm = np.asarray(truth[time_idx, :, :, target_indices].reshape(-1, len(target_indices)), dtype=np.float32)
            else:
                pred_denorm = np.asarray(pred_grid[rows, cols][:, target_indices], dtype=np.float32)
                target_denorm = np.asarray(state[time_idx, :, :][:, target_indices], dtype=np.float32)
            eval_time_idx = time_idx - lag
            update_reconstruction_metrics(metrics, pred_denorm, target_denorm, eval_time_idx, peak_range)

            if depth_output_position is not None:
                pred_depth = pred_denorm[:, depth_output_position]
                true_depth = target_denorm[:, depth_output_position]
                pred_inundated = pred_depth >= inundation_threshold
                true_inundated = true_depth >= inundation_threshold
                tp += int(np.count_nonzero(pred_inundated & true_inundated))
                fp += int(np.count_nonzero(pred_inundated & ~true_inundated))
                fn += int(np.count_nonzero(~pred_inundated & true_inundated))

                if metric_sensor_indices is not None:
                    sensor_indices = dense_sensor_indices if dense_sensor_indices is not None else metric_sensor_indices
                    usgs_pred.append(pred_depth[sensor_indices])
                    usgs_true.append(true_depth[sensor_indices])

        sensor_metrics = None
        if usgs_pred:
            pred_array = np.stack(usgs_pred, axis=0)
            true_array = np.stack(usgs_true, axis=0)
            sensor_metrics = []
            for sensor_pos, sensor_idx in enumerate(metric_sensor_indices):
                pred_series = pred_array[:, sensor_pos]
                true_series = true_array[:, sensor_pos]
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

        return {
            "trajectory_id": int(entry["traj_id"]),
            "prediction_path": str(prediction_path),
            "prediction_shape": pred_shape,
            "truth_path": str(truth_path) if truth_path is not None else str(root / entry["state_file"]) if entry.get("state_file") else str(root / entry["depth_file"]),
            "metric_accumulators": metrics,
            "metrics": summarize_reconstruction_metrics(metrics),
            "inundation_counts": {"tp": tp, "fp": fp, "fn": fn},
            "inundation_metrics": classification_metrics(tp, fp, fn) if depth_output_position is not None else None,
            "sensor_metrics": sensor_metrics,
        }
    finally:
        member.close()
        archive.close()


def write_chicago_baseline_results(output_path, prediction_dir, data_root, target_indices, lag, extent, results, aggregate, truth_mode, skipped_missing_truth=None):
    with open(output_path, "w") as file:
        file.write("\nBaseline Evaluation\n")
        file.write("===================\n")
        file.write(f"Prediction directory: {prediction_dir}\n")
        file.write(f"Data root: {data_root}\n")
        file.write(f"Target channels: {target_indices}\n")
        file.write("Baseline predictions are treated as denormalized outputs.\n")
        file.write(f"Truth mode: {truth_mode}\n")
        file.write(f"Predicted time range: lag..ntimes-1, lag={lag}\n")
        file.write("Dense-grid sparse mapping: dense flow_variables valid mask when available; otherwise historical create_dataset.py inverse coords formula\n")
        if skipped_missing_truth:
            file.write("Skipped trajectories without matching dense truth files:\n")
            for traj_id, truth_path in skipped_missing_truth:
                file.write(f"  trajectory {traj_id}: {truth_path}\n")
        for i in range(len(aggregate["metrics"]["mean_errors_denorm"])):
            channel = target_indices[i]
            file.write(f"MRE on U{channel} (denormalized, mean over time steps): {aggregate['metrics']['mean_errors_denorm'][i]}\n")
            file.write(f"Mean RMSE on U{channel} (denormalized, mean over time steps): {aggregate['metrics']['mean_rmse_denorm'][i]}\n")
            file.write(f"Relative prediction error on U{channel}: {aggregate['metrics']['rel_errors_denorm'][i]}\n")
        if target_indices and target_indices[0] == 0:
            file.write(f"MRE on U0 (denormalized, mean over time steps): {aggregate['metrics']['mre_denorm']}\n")
            file.write(f"Mean RMSE on U0 (denormalized, mean over time steps): {aggregate['metrics']['mean_rmse_u0_denorm']}\n")
            file.write(f"Relative prediction error on U0: {aggregate['metrics']['rel_error_denorm']}\n")
            file.write(f"Peak-period MRE on U0 (denormalized, mean over time steps): {aggregate['metrics']['peak_mre_denorm']}\n")
        if aggregate["inundation_metrics"] is not None:
            file.write("\nInundation Metrics Aggregated Over Time And Space\n")
            file.write("-------------------------------------------------\n")
            file.write(
                f"TP={aggregate['inundation_counts']['tp']}, FP={aggregate['inundation_counts']['fp']}, "
                f"FN={aggregate['inundation_counts']['fn']}\n"
            )
            file.write(f"CSI={aggregate['inundation_metrics']['csi']}\n")
            file.write(f"F1={aggregate['inundation_metrics']['f1']}\n")
            file.write(f"Precision={aggregate['inundation_metrics']['precision']}\n")
            file.write(f"Recall={aggregate['inundation_metrics']['recall']}\n")

        file.write("\nPer-Trajectory Results\n")
        file.write("----------------------\n")
        for result in results:
            file.write(f"\nTrajectory id: {result['trajectory_id']}\n")
            file.write(f"Prediction path: {result['prediction_path']}\n")
            file.write(f"Prediction shape: {result['prediction_shape']}\n")
            file.write(f"Truth path: {result['truth_path']}\n")
            for i in range(len(result["metrics"]["mean_errors_denorm"])):
                channel = target_indices[i]
                file.write(f"MRE on U{channel} (denormalized, mean over time steps): {result['metrics']['mean_errors_denorm'][i]}\n")
                file.write(f"Mean RMSE on U{channel} (denormalized, mean over time steps): {result['metrics']['mean_rmse_denorm'][i]}\n")
                file.write(f"Relative prediction error on U{channel}: {result['metrics']['rel_errors_denorm'][i]}\n")
            if target_indices and target_indices[0] == 0:
                file.write(f"MRE on U0 (denormalized, mean over time steps): {result['metrics']['mre_denorm']}\n")
                file.write(f"Mean RMSE on U0 (denormalized, mean over time steps): {result['metrics']['mean_rmse_u0_denorm']}\n")
                file.write(f"Relative prediction error on U0: {result['metrics']['rel_error_denorm']}\n")
                file.write(f"Peak-period MRE on U0 (denormalized, mean over time steps): {result['metrics']['peak_mre_denorm']}\n")
            if result["inundation_metrics"] is not None:
                file.write("Inundation Metrics Aggregated Over Time And Space\n")
                file.write(
                    f"TP={result['inundation_counts']['tp']}, FP={result['inundation_counts']['fp']}, "
                    f"FN={result['inundation_counts']['fn']}\n"
                )
                file.write(f"CSI={result['inundation_metrics']['csi']}\n")
                file.write(f"F1={result['inundation_metrics']['f1']}\n")
                file.write(f"Precision={result['inundation_metrics']['precision']}\n")
                file.write(f"Recall={result['inundation_metrics']['recall']}\n")
            if result["sensor_metrics"] is not None:
                file.write("USGS Sensor Time-Series Metrics\n")
                for item in result["sensor_metrics"]:
                    file.write(
                        f"Sensor {item['sensor_position']} (grid_index={item['grid_index']}): "
                        f"NSE={item['nse']}, KGE={item['kge']}, "
                        f"peak_depth_relative_error={item['peak_depth_relative_error']}, "
                        f"true_peak_depth={item['true_peak_depth']}, "
                        f"predicted_peak_depth={item['predicted_peak_depth']}\n"
                    )


def resolve_prediction_files(prediction_dir, trajectory_ids):
    prediction_dir = Path(prediction_dir)
    if trajectory_ids is None:
        trajectory_ids = []
        for path in sorted(prediction_dir.glob("predictions_traj*.npz")):
            stem = path.stem
            trajectory_ids.append(int(stem.replace("predictions_traj", "")))
    files = [(traj_id, prediction_dir / PREDICTION_PATTERN.format(traj_id=traj_id)) for traj_id in trajectory_ids]
    missing = [str(path) for _, path in files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing baseline prediction files:\n" + "\n".join(missing))
    return files




def filter_prediction_files_for_dense_truth(prediction_files, truth_dir, explicit_trajectory_ids, allow_sparse_truth):
    if truth_dir is None or allow_sparse_truth:
        return prediction_files, []
    available = []
    missing = []
    for traj_id, prediction_path in prediction_files:
        truth_path = Path(truth_dir) / f"truth_traj{traj_id}.npy"
        if truth_path.exists():
            available.append((traj_id, prediction_path))
        else:
            missing.append((traj_id, truth_path))
    if missing and explicit_trajectory_ids:
        details = "\n".join(f"trajectory {traj_id}: {truth_path}" for traj_id, truth_path in missing)
        raise FileNotFoundError("Missing requested dense truth files:\n" + details)
    if not available:
        details = "\n".join(f"trajectory {traj_id}: {truth_path}" for traj_id, truth_path in missing)
        raise FileNotFoundError("No prediction files have matching dense truth files:\n" + details)
    return available, missing


def load_eval_config(config_path):
    if config_path is None:
        return {}
    path = Path(config_path)
    if path.is_dir():
        path = path / "config.yaml"
    with open(path, "r") as handle:
        config = yaml.safe_load(handle) or {}
    return config.get("basics", {}) or {}


def resolve_eval_options(args):
    basics = load_eval_config(args.config)
    data_root = args.data_root if args.data_root is not None else basics.get("data_root", "data/Chicago")
    target_variables = args.target_variables if args.target_variables is not None else basics.get("target_variables", "depth")
    lag = args.lag if args.lag is not None else int(basics.get("lag", 12))
    return data_root, target_variables, lag


def filter_prediction_files_for_manifest_test_set(prediction_files, entries):
    _, _, idx_test = chicago_split_indices(entries)
    test_ids = {int(entries[idx]["traj_id"]) for idx in idx_test}
    return [(traj_id, path) for traj_id, path in prediction_files if int(traj_id) in test_ids]


def main(args):
    data_root, target_variables, lag = resolve_eval_options(args)
    root, manifest, entries = load_chicago_manifest(data_root)
    ntimes = int(manifest["time_steps"])
    if lag < 0 or lag >= ntimes:
        raise ValueError(f"lag must satisfy 0 <= lag < {ntimes}, got {lag}")

    dataset_output_size = get_chicago_output_size(root, entries[0])
    target_indices = resolve_chicago_targets({"basics": {"target_variables": target_variables}}, dataset_output_size)
    entries_by_id = {int(entry["traj_id"]): entry for entry in entries}

    prediction_files = resolve_prediction_files(args.prediction_dir, args.trajectory_ids)
    if args.only_manifest_test_set:
        prediction_files = filter_prediction_files_for_manifest_test_set(prediction_files, entries)
        if not prediction_files:
            raise ValueError("No baseline prediction files match the C-STRIDE Chicago test split.")

    use_sparse_truth = args.truth_mode == "sparse" or args.allow_sparse_truth
    truth_dir = None if use_sparse_truth else args.truth_dir
    if not use_sparse_truth and truth_dir is None:
        raise ValueError("--truth_dir is required when --truth_mode dense")

    prediction_files, skipped_missing_truth = filter_prediction_files_for_dense_truth(
        prediction_files,
        truth_dir,
        explicit_trajectory_ids=args.trajectory_ids is not None,
        allow_sparse_truth=use_sparse_truth,
    )
    for traj_id, truth_path in skipped_missing_truth:
        print(f"Skipping trajectory {traj_id}: missing dense truth {truth_path}")

    results = []
    for traj_id, prediction_path in prediction_files:
        if traj_id not in entries_by_id:
            raise ValueError(f"Trajectory id {traj_id} is not present in the Chicago manifest.")
        metric_sensor_indices = None
        if args.usgs_sensor_path and 0 in target_indices:
            metric_sensor_indices = load_metric_sensor_indices(args.usgs_sensor_path, entries, root, entries_by_id[traj_id])
        result = evaluate_baseline_trajectory(
            prediction_path=prediction_path,
            entry=entries_by_id[traj_id],
            root=root,
            target_indices=target_indices,
            extent=args.extent,
            truth_dir=truth_dir,
            allow_sparse_truth=use_sparse_truth,
            member_name=args.member_name,
            lag=lag,
            inundation_threshold=args.inundation_threshold,
            metric_sensor_indices=metric_sensor_indices,
            source_dense_roots=args.source_dense_roots,
            valid_mask_mode=args.valid_mask_mode,
            burn_in=args.burn_in_length,
            depth_threshold=args.depth_threshold,
        )
        results.append(result)
        print(f"Evaluated trajectory {traj_id}: {prediction_path}")

    aggregate_accumulators = merge_reconstruction_metrics([result["metric_accumulators"] for result in results])
    aggregate_counts = {
        "tp": sum(result["inundation_counts"]["tp"] for result in results),
        "fp": sum(result["inundation_counts"]["fp"] for result in results),
        "fn": sum(result["inundation_counts"]["fn"] for result in results),
    }
    aggregate = {
        "metrics": summarize_reconstruction_metrics(aggregate_accumulators),
        "inundation_counts": aggregate_counts,
        "inundation_metrics": classification_metrics(**aggregate_counts) if 0 in target_indices else None,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_chicago_baseline_results(
        output_path=output_path,
        prediction_dir=args.prediction_dir,
        data_root=root,
        target_indices=target_indices,
        lag=lag,
        extent=args.extent,
        results=results,
        aggregate=aggregate,
        truth_mode="sparse" if use_sparse_truth else "dense",
        skipped_missing_truth=skipped_missing_truth,
    )
    print(f"Baseline evaluation written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate denormalized full-grid Chicago baseline predictions")
    parser.add_argument("--prediction_dir", required=True)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/next_step/full.yaml"),
        help="C-STRIDE config or run directory used to mirror data_root, lag, and target_variables.",
    )
    parser.add_argument("--data_root", default=None, help="Override the config data_root.")
    parser.add_argument("--output", default="baseline_chicago_results.txt")
    parser.add_argument("--trajectory_ids", type=int, nargs="*", default=None)
    parser.add_argument(
        "--only_manifest_test_set",
        action="store_true",
        default=True,
        help="Restrict scanned prediction files to the same manifest test split used by evaluate_next_step.py. Enabled by default.",
    )
    parser.add_argument(
        "--include_non_test_predictions",
        dest="only_manifest_test_set",
        action="store_false",
        help="Evaluate every prediction file found, including trajectories outside the C-STRIDE test split.",
    )
    parser.add_argument("--target_variables", default=None, help="Override config target_variables: 'depth', 'all', or channel indices")
    parser.add_argument("--member_name", default="pred_full.npy")
    parser.add_argument("--lag", type=int, default=None, help="Override config lag; snapshots before lag are skipped.")
    parser.add_argument("--inundation_threshold", type=float, default=0.5)
    parser.add_argument("--burn_in_length", type=int, default=1)
    parser.add_argument("--depth_threshold", type=float, default=0.1)
    parser.add_argument(
        "--valid_mask_mode",
        choices=("auto", "dense_flow", "coords"),
        default="auto",
        help="How to map sparse truth cells to the dense prediction grid. auto uses dense flow_variables.npy masks when found.",
    )
    parser.add_argument(
        "--source_dense_roots",
        nargs="*",
        default=[],
        help="Roots containing event_*/flow_variables.npy used to reconstruct exact valid-cell masks.",
    )
    parser.add_argument("--usgs_sensor_path", default=str(ROOT / "data/gauges/USGS_6sensors.json"))
    parser.add_argument("--truth_mode", choices=("sparse", "dense"), default="sparse")
    parser.add_argument("--truth_dir", default=None)
    parser.add_argument(
        "--allow_sparse_truth",
        action="store_true",
        help="Backward-compatible alias for --truth_mode sparse.",
    )
    parser.add_argument(
        "--extent",
        type=float,
        nargs=4,
        default=(-1.0, 1.0, -3.052734375, 3.052734375),
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
        help="Deprecated; sparse mapping now follows historical create_dataset.py coordinate construction.",
    )
    main(parser.parse_args())
