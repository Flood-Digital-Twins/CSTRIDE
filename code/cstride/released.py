"""Prediction from compact released inputs, without simulation fields."""
from pathlib import Path
import numpy as np
import torch
import yaml
from .runtime import ROOT, load_state_dict
from .chicago import load_sensor_map_file, load_chicago_normalizers, chicago_include_precipitation_input, chicago_use_terrain_conditioning, make_windows
from .chicago_forecast import build_chicago_feature_coords
from .models import CSTRIDE
from .forecast_observations import SensorObservationLSTM


def load_inputs(model_name, storm, cldnet_root=None, observations="simulated", device="cpu", auxiliary_rainfall=False):
    run = ROOT / "checkpoints" / model_name
    config = yaml.safe_load((run / "config.yaml").read_text())
    norms = load_chicago_normalizers(ROOT / "data/normalization/normalizers.npz")
    sensors = load_sensor_map_file(run / "sensor_indices.json")[storm]
    six = load_sensor_map_file(ROOT / "data/gauges/USGS_6sensors.json")[storm]
    positions = [int(np.flatnonzero(six == cell)[0]) for cell in sensors]
    with np.load(ROOT / "data/gauges/simulated_depth.npz", allow_pickle=False) as gauges:
        matches = np.flatnonzero(gauges["trajectory_ids"] == storm)
        if len(matches) != 1:
            raise ValueError(f"No unique gauge history for storm {storm}")
        depth = gauges["depth_m"][matches[0]][:, positions]
    if observations == "usgs":
        if storm != 120:
            raise ValueError("USGS inputs are available only for storm 120")
        with np.load(ROOT / "data/gauges/usgs_2013/usgs_observations.npz", allow_pickle=False) as record:
            depth = record["input_depth"][:, positions]
    gauge_sequence = norms["depth"].transform(depth)
    rain = None
    if chicago_include_precipitation_input(config) or auxiliary_rainfall:
        external = Path(cldnet_root) if cldnet_root else ROOT.parent / "cldnet"
        event = "event_2013-04-17_2013-04-21" if storm == 120 else f"event_{storm}"
        path = external / "data/forcings" / event / "rain_source.npy"
        rain = np.load(path, allow_pickle=False)[:96].reshape(96, -1).astype(np.float32).astype(np.float16)
        rain = norms["precip"].transform(rain)
    sequence = np.concatenate([gauge_sequence, rain], axis=-1) if chicago_include_precipitation_input(config) else gauge_sequence
    with np.load(ROOT / f"data/grids/grid_traj{storm}.npz", allow_pickle=False) as grid:
        coords = build_chicago_feature_coords(grid["coords"], grid["elevation"], grid["manning"], grid["slope"], norms, chicago_use_terrain_conditioning(config))
    model = CSTRIDE(input_size=sequence.shape[-1], output_size=1, coord_size=coords.shape[-1], device=torch.device(device), **config["model"]).to(device)
    model.load_state_dict(load_state_dict(run / "best_model.ckpt", device))
    model.eval()
    return model, config, norms, torch.tensor(sequence, dtype=torch.float32), coords, rain, positions


def encode(model, history):
    _, (hidden, _) = model.lstm(history)
    return hidden[-1]


def decode_chunks(model, latent, coords, chunk_size):
    if chunk_size < 1:
        raise ValueError("chunk-size must be positive")
    for start in range(0, len(coords), chunk_size):
        query = torch.tensor(coords[start:start+chunk_size], device=latent.device).unsqueeze(0)
        yield start, model.decode(query, latent)[0, :, 0].cpu().numpy()


def load_gauge_forecaster(model_name, rainfall, device):
    name = "sensor_observation_lstm_precip.ckpt" if rainfall else "sensor_observation_lstm.ckpt"
    path = ROOT / "checkpoints" / model_name / name
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = SensorObservationLSTM(**{k: checkpoint[k] for k in ("input_size", "sensor_size", "hidden_size", "num_layers", "dropout")}).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
