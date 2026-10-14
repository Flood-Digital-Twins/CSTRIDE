from .runtime import ROOT, load_config, parse_args, load_state_dict
import argparse
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .chicago import (
    ChicagoTrajectoryDataset,
    add_chicago_input_noise,
    apply_chicago_sensor_dropout,
    build_sensor_sequences,
    chicago_include_precipitation_input,
    chicago_split_indices,
    chicago_use_terrain_conditioning,
    fit_chicago_normalizers,
    get_chicago_output_size,
    get_chicago_target_normalizer,
    load_chicago_manifest,
    load_chicago_sensors,
    extend_chicago_sensor_map,
    make_windows,
    resolve_chicago_observables,
    resolve_chicago_targets,
)
from .evaluate import get_model_params
from .models import CSTRIDE
from .processdata import num2p


def compute_denormalized_metrics(model, dataloader, device, state_normalizer, nt_valid):
    total_gt_norm = 0.0
    total_err_norm = 0.0
    sample_rel_norm = []
    total_gt_denorm = None
    total_err_denorm = None
    sample_rel_denorm = None
    sample_rmse_denorm = None
    peak_rel_denorm = []

    peak_start_orig = 48
    peak_end_orig = 84
    peak_start_nop = peak_start_orig - dataloader.dataset.lag
    peak_end_nop = peak_end_orig - dataloader.dataset.lag

    sample_idx = 0
    model.eval()
    with torch.no_grad():
        for batch_x, batch_coords, batch_target in dataloader:
            batch_x = batch_x.to(device)
            batch_coords = batch_coords.to(device)
            batch_target = batch_target.to(device)
            pred, _ = model(batch_x, batch_coords)

            total_gt_norm += torch.sum(batch_target ** 2).item()
            total_err_norm += torch.sum((pred - batch_target) ** 2).item()

            pred_np = pred.detach().cpu().numpy()[0]
            target_np = batch_target.detach().cpu().numpy()[0]
            sample_gt_norm = np.sum(target_np ** 2)
            sample_err_norm = np.sum((pred_np - target_np) ** 2)
            if sample_gt_norm > 0:
                sample_rel_norm.append(np.sqrt(sample_err_norm / sample_gt_norm))

            pred_denorm = state_normalizer.inverse_transform(pred_np)
            target_denorm = state_normalizer.inverse_transform(target_np)

            output_size = pred_denorm.shape[-1]
            if total_gt_denorm is None:
                total_gt_denorm = np.zeros(output_size, dtype=np.float64)
                total_err_denorm = np.zeros(output_size, dtype=np.float64)
                sample_rel_denorm = [[] for _ in range(output_size)]
                sample_rmse_denorm = [[] for _ in range(output_size)]

            for channel_idx in range(output_size):
                sq_err = (pred_denorm[:, channel_idx] - target_denorm[:, channel_idx]) ** 2
                err = np.sum(sq_err)
                gt = np.sum(target_denorm[:, channel_idx] ** 2)
                total_err_denorm[channel_idx] += err
                total_gt_denorm[channel_idx] += gt
                sample_rmse_denorm[channel_idx].append(float(np.sqrt(np.mean(sq_err))))
                if gt > 0:
                    sample_rel_denorm[channel_idx].append(np.sqrt(err / gt))

            time_idx = sample_idx % nt_valid
            peak_gt = np.sum(target_denorm[:, 0] ** 2)
            peak_err = np.sum((pred_denorm[:, 0] - target_denorm[:, 0]) ** 2)
            if peak_start_nop <= time_idx < peak_end_nop and peak_gt > 0:
                peak_rel_denorm.append(np.sqrt(peak_err / peak_gt))
            sample_idx += 1

    return {
        "rel_error_norm": np.sqrt(total_err_norm / total_gt_norm),
        "mean_error_norm": num2p(float(np.mean(sample_rel_norm))) if sample_rel_norm else "nan%",
        "mean_errors_denorm": [
            num2p(float(np.mean(channel_errors))) if channel_errors else "nan%"
            for channel_errors in sample_rel_denorm
        ],
        "mean_rmse_denorm": [
            float(np.mean(channel_rmse)) if channel_rmse else float("nan")
            for channel_rmse in sample_rmse_denorm
        ],
        "rel_errors_denorm": [
            float(np.sqrt(err / gt)) if gt > 0 else float("nan")
            for err, gt in zip(total_err_denorm, total_gt_denorm)
        ],
        "mre_denorm": num2p(float(np.mean(sample_rel_denorm[0]))) if sample_rel_denorm[0] else "nan%",
        "mean_rmse_u0_denorm": float(np.mean(sample_rmse_denorm[0]))
        if sample_rmse_denorm[0]
        else float("nan"),
        "rel_error_denorm": float(np.sqrt(total_err_denorm[0] / total_gt_denorm[0]))
        if total_gt_denorm[0] > 0
        else float("nan"),
        "peak_mre_denorm": num2p(float(np.mean(peak_rel_denorm))) if peak_rel_denorm else "nan%",
    }


def save_chicago_evaluation_results(path, total_params, trainable_params, metrics, sensor_dropout_info):
    with open(os.path.join(path, "results.txt"), "a") as file:
        file.write("\nEvaluation\n")
        file.write("================\n")
        file.write(f"Total parameters in the model: {total_params}\n")
        file.write(f"Trainable parameters in the model: {trainable_params}\n")
        file.write(f"Sensor dropout: {sensor_dropout_info}\n")
        file.write(f"Relative prediction error on U (Normalized): {metrics['rel_error_norm']}\n")
        file.write(f"Mean relative prediction error on U (Normalized): {metrics['mean_error_norm']}\n")
        for i in range(len(metrics["mean_errors_denorm"])):
            file.write(f"MRE on U{i} (denormalized, mean over time steps): {metrics['mean_errors_denorm'][i]}\n")
            file.write(f"Mean RMSE on U{i} (denormalized, mean over time steps): {metrics['mean_rmse_denorm'][i]}\n")
            file.write(f"Relative prediction error on U{i}: {metrics['rel_errors_denorm'][i]}\n")
        file.write(f"MRE on U0 (denormalized, mean over time steps): {metrics['mre_denorm']}\n")
        file.write(f"Mean RMSE on U0 (denormalized, mean over time steps): {metrics['mean_rmse_u0_denorm']}\n")
        file.write(f"Relative prediction error on U0: {metrics['rel_error_denorm']}\n")
        file.write(f"Peak-period MRE on U0 (denormalized, mean over time steps): {metrics['peak_mre_denorm']}\n")


def main(path):
    with open(os.path.join(path, "config.yaml"), "r") as f:
        config = load_config(f)

    if config["basics"].get("scaler", "global") != "global":
        raise ValueError("Chicago evaluation currently supports only global scaling.")

    root, manifest, entries = load_chicago_manifest(config["basics"].get("data_root", "data/Chicago"))
    ntrajectories = len(entries)
    ntimes = int(manifest["time_steps"])
    lag = config["basics"]["lag"]
    nt_valid = ntimes - lag
    include_precipitation = chicago_include_precipitation_input(config)
    terrain_conditioning = chicago_use_terrain_conditioning(config)
    coord_size = 5 if terrain_conditioning else 2
    dataset_output_size = get_chicago_output_size(root, entries[0])
    observable_indices = resolve_chicago_observables(config, dataset_output_size)
    target_indices = resolve_chicago_targets(config, dataset_output_size)
    output_size = len(target_indices)
    num_sensors = int(config["basics"].get("num_sensors", 0))
    if num_sensors == 0 and not include_precipitation:
        raise ValueError(
            "Chicago evaluation requires at least one temporal input feature. "
            "Set num_sensors > 0 or include_precipitation_input: true."
        )

    idx_train, _, idx_test = chicago_split_indices(entries, config["basics"].get("split_file"))
    normalizers = fit_chicago_normalizers(entries, idx_train, root)
    target_normalizer = get_chicago_target_normalizer(normalizers, target_indices)
    sensor_map = load_chicago_sensors(path, config)
    if sensor_map is None:
        raise FileNotFoundError("No saved Chicago sensor mapping found in run folder.")
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
    sensor_feature_count = num_sensors * len(observable_indices)
    if config["basics"].get("noise", 0.0) > 0.0:
        sensors_data = add_chicago_input_noise(
            sensors_data,
            config["basics"]["noise"],
            sensor_feature_count=sensor_feature_count,
        )
    sensors_data, sensor_dropout_info = apply_chicago_sensor_dropout(
        sensors_data,
        config,
        num_sensors=num_sensors,
        observable_count=len(observable_indices),
        sensor_map=sensor_map,
        entries=entries,
        root=root,
    )
    test_windows = make_windows(sensors_data[idx_test], lag)

    test_dataset = ChicagoTrajectoryDataset(
        entries=[entries[i] for i in idx_test],
        root=root,
        input_windows=test_windows,
        nt=nt_valid,
        lag=lag,
        normalizers=normalizers,
        target_indices=target_indices,
        terrain_conditioning=terrain_conditioning,
        num_points=None,
        spatial_probs=None,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["training"].get("num_workers", 4),
        pin_memory=True,
        persistent_workers=config["training"].get("num_workers", 8) > 0,
    )

    device = torch.device(config["basics"]["device"] if torch.cuda.is_available() else "cpu")
    model = CSTRIDE(
        input_size=sensors_data.shape[-1],
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
    state_dict = load_state_dict(ckpt_path, device)
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    # model = torch.compile(model) if config["training"].get("compile", False) else model

    total_params, trainable_params = get_model_params(model)
    metrics = compute_denormalized_metrics(model, test_loader, device, target_normalizer, nt_valid)

    if num_sensors == 0:
        temporal_mode = "precip-only" if include_precipitation else "invalid"
    elif include_precipitation:
        temporal_mode = "sensor+precip"
    else:
        temporal_mode = "sensor-only"
    print(f"Chicago evaluation input mode: {temporal_mode}, terrain_conditioning={terrain_conditioning}")
    print(f"Sensor dropout: {sensor_dropout_info}")
    print(f"Total parameters in the model: {total_params}")
    print(f"Trainable parameters in the model: {trainable_params}")
    print(f"Relative prediction error on U (Normalized): {metrics['rel_error_norm']}")
    print(f"Mean relative prediction error on U (Normalized): {metrics['mean_error_norm']}")
    for i in range(output_size):
        print(f"MRE on U{i} (denormalized, mean over time steps): {metrics['mean_errors_denorm'][i]}")
        print(f"Mean RMSE on U{i} (denormalized, mean over time steps): {metrics['mean_rmse_denorm'][i]}")
        print(f"Relative prediction error on U{i}: {metrics['rel_errors_denorm'][i]}")
    print(f"MRE on U0 (denormalized, mean over time steps): {metrics['mre_denorm']}")
    print(f"Mean RMSE on U0 (denormalized, mean over time steps): {metrics['mean_rmse_u0_denorm']}")
    print(f"Relative prediction error on U0: {metrics['rel_error_denorm']}")
    print(f"Peak-period MRE on U0 (denormalized, mean over time steps): {metrics['peak_mre_denorm']}")

    save_chicago_evaluation_results(path, total_params, trainable_params, metrics, sensor_dropout_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate C-STRIDE on the Chicago dataset")
    parser.add_argument("--path", required=True)
    args = parse_args(parser)
    main(args.path)
