from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .chicago import get_chicago_state_array, get_chicago_target_normalizer
from .processdata import Padding


def build_chicago_feature_coords(coords, elevation, manning, slope, normalizers, terrain_conditioning=True):
    coords = np.asarray(coords, dtype=np.float32)
    if not terrain_conditioning:
        return coords

    elevation = normalizers["elevation"].transform(elevation)[:, None]
    manning = normalizers["manning"].transform(manning)[:, None]
    slope = normalizers["slope"].transform(slope)[:, None]
    return np.concatenate([coords, elevation, manning, slope], axis=-1).astype(np.float32)


def build_chicago_sensor_coord_sequences(entries, root, sensor_map, normalizers):
    root = Path(root)
    sequences = []
    for entry in entries:
        traj_id = entry["traj_id"]
        sensor_idx = sensor_map[traj_id]
        coords = np.load(root / entry["coords_file"], mmap_mode="r")[sensor_idx]
        elevation = np.load(root / entry["elevation_file"], mmap_mode="r")[sensor_idx]
        manning = np.load(root / entry["manning_file"], mmap_mode="r")[sensor_idx]
        slope = np.load(root / entry["slope_file"], mmap_mode="r")[sensor_idx]
        sensor_coords = build_chicago_feature_coords(coords, elevation, manning, slope, normalizers)
        ntimes = get_chicago_state_array(root, entry).shape[0]
        sequences.append(torch.tensor(sensor_coords, dtype=torch.float32).unsqueeze(0).repeat(ntimes, 1, 1))
    return sequences


def make_forecast_histories(sensors_data, lag, horizon):
    if sensors_data.dim() != 3:
        raise ValueError("sensors_data must have shape (ntraj, ntimes, nfeatures).")
    ntraj, ntimes, _ = sensors_data.shape
    valid_times = ntimes - horizon
    if valid_times <= 0:
        raise ValueError(f"Forecast horizon {horizon} is too large for ntimes={ntimes}.")
    histories = Padding(sensors_data, lag).reshape(ntraj, ntimes, lag, -1)
    return histories[:, :valid_times].reshape(ntraj * valid_times, lag, -1)


def make_future_exogenous(entries, root, normalizers, horizon, max_steps=None, include_precipitation=True):
    root = Path(root)
    all_future = []
    for entry in entries:
        if not include_precipitation:
            ntimes = int(max_steps) if max_steps is not None else get_chicago_state_array(root, entry).shape[0]
            valid_times = ntimes - horizon
            if valid_times <= 0:
                raise ValueError(f"Forecast horizon {horizon} is too large for ntimes={ntimes}.")
            all_future.append(torch.empty((valid_times, horizon, 0), dtype=torch.float32))
            continue

        precip = np.load(root / entry["precip_file"], mmap_mode="r")
        precip = normalizers["precip"].transform(precip)
        if max_steps is not None:
            precip = precip[:max_steps]
        valid_times = precip.shape[0] - horizon
        future = np.stack(
            [precip[step + 1 : step + 1 + valid_times] for step in range(horizon)],
            axis=1,
        )
        all_future.append(torch.tensor(future, dtype=torch.float32))
    return torch.cat(all_future, dim=0)


class ChicagoForecastWindowDataset(Dataset):
    def __init__(
        self,
        entries,
        root,
        input_windows,
        future_exogenous,
        nt,
        horizon,
        normalizers,
        target_indices,
        sensor_map,
        terrain_conditioning=True,
        num_points=None,
        spatial_probs=None,
    ):
        super().__init__()
        self.entries = entries
        self.root = Path(root)
        self.x_data = input_windows.float().cpu()
        self.future_exogenous = future_exogenous.float().cpu()
        self.nt = int(nt)
        self.horizon = int(horizon)
        self.normalizers = normalizers
        self.target_indices = list(target_indices)
        self.target_normalizer = get_chicago_target_normalizer(normalizers, self.target_indices)
        self.sensor_map = sensor_map
        self.terrain_conditioning = bool(terrain_conditioning)
        self.num_points = num_points
        self.spatial_probs = spatial_probs or {}

        self.state_arrays = [get_chicago_state_array(self.root, entry) for entry in entries]
        self.coords_arrays = [np.load(self.root / entry["coords_file"], mmap_mode="r") for entry in entries]
        if self.terrain_conditioning:
            self.elev_arrays = [np.load(self.root / entry["elevation_file"], mmap_mode="r") for entry in entries]
            self.manning_arrays = [np.load(self.root / entry["manning_file"], mmap_mode="r") for entry in entries]
            self.slope_arrays = [np.load(self.root / entry["slope_file"], mmap_mode="r") for entry in entries]
        else:
            self.elev_arrays = None
            self.manning_arrays = None
            self.slope_arrays = None

        self.num_trajectories = len(entries)
        self.num_samples = self.num_trajectories * self.nt
        if self.x_data.shape[0] != self.num_samples:
            raise ValueError("input_windows sample count does not match entries * nt.")
        if self.future_exogenous.shape[:2] != (self.num_samples, self.horizon):
            raise ValueError("future_exogenous must have shape (num_samples, horizon, forcing_dim).")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        traj_idx = idx // self.nt
        time_idx = idx % self.nt
        entry = self.entries[traj_idx]

        coords = self.coords_arrays[traj_idx]
        if self.terrain_conditioning:
            elevation = self.elev_arrays[traj_idx]
            manning = self.manning_arrays[traj_idx]
            slope = self.slope_arrays[traj_idx]
        else:
            elevation = None
            manning = None
            slope = None

        if self.num_points is not None:
            nstate = coords.shape[0]
            sample_count = min(self.num_points, nstate)
            probs = self.spatial_probs.get(entry["traj_id"])
            indices = np.random.choice(nstate, size=sample_count, replace=False, p=probs)
            coords = coords[indices]
            if self.terrain_conditioning:
                elevation = elevation[indices]
                manning = manning[indices]
                slope = slope[indices]
        else:
            indices = slice(None)

        feature_coords = build_chicago_feature_coords(
            coords,
            elevation,
            manning,
            slope,
            self.normalizers,
            terrain_conditioning=self.terrain_conditioning,
        )
        future_coords = np.repeat(feature_coords[None, :, :], self.horizon, axis=0)

        future_targets = []
        for step in range(1, self.horizon + 1):
            state_t = self.state_arrays[traj_idx][time_idx + step, indices]
            target = state_t[:, self.target_indices]
            future_targets.append(self.target_normalizer.transform(target).astype(np.float32))
        future_targets = np.stack(future_targets, axis=0)

        sensor_indices = self.sensor_map[entry["traj_id"]]
        if self.terrain_conditioning:
            sensor_coords = build_chicago_feature_coords(
                self.coords_arrays[traj_idx][sensor_indices],
                self.elev_arrays[traj_idx][sensor_indices],
                self.manning_arrays[traj_idx][sensor_indices],
                self.slope_arrays[traj_idx][sensor_indices],
                self.normalizers,
                terrain_conditioning=True,
            )
        else:
            sensor_coords = build_chicago_feature_coords(
                self.coords_arrays[traj_idx][sensor_indices],
                None,
                None,
                None,
                self.normalizers,
                terrain_conditioning=False,
            )
        future_sensor_coords = np.repeat(sensor_coords[None, :, :], self.horizon, axis=0)

        return (
            self.x_data[idx],
            torch.tensor(future_coords, dtype=torch.float32),
            torch.tensor(future_targets, dtype=torch.float32),
            torch.tensor(future_sensor_coords, dtype=torch.float32),
            self.future_exogenous[idx],
        )
