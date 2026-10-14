import json
import os

from .runtime import ROOT
import math
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MinMaxNormalizer:
    def __init__(self, min_value, max_value):
        self.min_value = np.asarray(min_value, dtype=np.float32)
        self.max_value = np.asarray(max_value, dtype=np.float32)
        self.scale = self.max_value - self.min_value
        self.scale = np.where(self.scale == 0, 1.0, self.scale).astype(np.float32)

    def transform(self, values):
        values = np.asarray(values, dtype=np.float32)
        return 2.0 * (values - self.min_value) / self.scale - 1.0

    def inverse_transform(self, values):
        values = np.asarray(values, dtype=np.float32)
        return 0.5 * (values + 1.0) * self.scale + self.min_value


def _load_state_file(path):
    state = np.load(path, mmap_mode="r")
    if state.ndim == 4:
        if state.shape[0] != 1:
            raise ValueError(f"Expected leading singleton batch dimension in {path}, got shape {state.shape}")
        state = state[0]
    if state.ndim != 3:
        raise ValueError(f"Expected Chicago state tensor with shape (time, points, channels), got {state.shape}")
    return state


def load_chicago_manifest(root="data/Chicago"):
    root = Path(os.environ.get("CSTRIDE_DATA_ROOT", root))
    if not root.is_absolute():
        root = ROOT / root
    manifest = json.loads((root / "manifest.json").read_text())
    entries = sorted(manifest["files"], key=lambda item: item["traj_id"])
    return root, manifest, entries


def get_chicago_state_array(root, entry):
    root = Path(root)
    if entry.get("state_file"):
        return _load_state_file(root / entry["state_file"])
    return np.load(root / entry["depth_file"], mmap_mode="r")


def get_chicago_output_size(root, entry):
    state = get_chicago_state_array(root, entry)
    return int(state.shape[-1])


def resolve_chicago_observables(config, output_size):
    observables = config["basics"].get("observables")
    if observables is None:
        return [0]
    return _resolve_channel_indices(observables, output_size, "observable")


def resolve_chicago_targets(config, output_size):
    targets = config["basics"].get("target_variables", "depth")
    if targets == "depth":
        return [0]
    if targets == "all":
        return list(range(output_size))
    return _resolve_channel_indices(targets, output_size, "target")


def get_chicago_target_normalizer(normalizers, target_indices):
    if len(target_indices) == 1 and target_indices[0] == 0:
        return normalizers["depth"]
    state_normalizer = normalizers["state"]
    return MinMaxNormalizer(state_normalizer.min_value[target_indices], state_normalizer.max_value[target_indices])


def chicago_include_precipitation_input(config):
    return bool(config["basics"].get("include_precipitation_input", True))


def chicago_use_terrain_conditioning(config):
    return bool(config["basics"].get("terrain_conditioning", True))


def add_chicago_input_noise(sensors_data, noise_level, sensor_feature_count):
    if noise_level <= 0.0:
        return sensors_data

    if sensors_data.ndim < 2:
        raise ValueError(f"Expected sensors_data with at least 2 dims, got shape {tuple(sensors_data.shape)}")

    total_feature_count = int(sensors_data.shape[-1])
    sensor_feature_count = int(sensor_feature_count)
    if sensor_feature_count < 0 or sensor_feature_count > total_feature_count:
        raise ValueError(
            f"sensor_feature_count={sensor_feature_count} is invalid for total_feature_count={total_feature_count}"
        )

    noisy = sensors_data.clone()
    if sensor_feature_count > 0:
        sensor_block = noisy[..., :sensor_feature_count]
        sensor_std = noise_level * torch.std(sensor_block, dim=-1, keepdim=True)
        noisy[..., :sensor_feature_count] = sensor_block + sensor_std * torch.randn_like(sensor_block)

    if sensor_feature_count < total_feature_count:
        precip_block = noisy[..., sensor_feature_count:]
        precip_std = noise_level * torch.std(precip_block, dim=-1, keepdim=True)
        noisy[..., sensor_feature_count:] = precip_block + precip_std * torch.randn_like(precip_block)

    return noisy


def chicago_sensor_dropout_settings(config):
    basics = config["basics"]
    dropout_config = config.get("sensor_dropout", {})
    rate = float(dropout_config.get("rate", basics.get("sensor_dropout_rate", 0.0)))
    always_available = dropout_config.get(
        "always_available_sensors",
        basics.get("sensor_dropout_always_available_sensors", 4),
    )
    always_available_path = dropout_config.get(
        "always_available_sensor_path",
        basics.get("sensor_dropout_always_available_sensor_path"),
    )
    if always_available_path is None and isinstance(always_available, str):
        always_available_path = always_available
        always_available = None
    if always_available is None:
        always_available = 0
    seed = dropout_config.get("seed", basics.get("sensor_dropout_seed"))
    if seed is None:
        seed = int(basics.get("random_seed", 0))
    return rate, int(always_available), always_available_path, int(seed)


def _load_sensor_dropout_map(path):
    sensor_obj = load_sensor_map_file(path)
    if isinstance(sensor_obj, dict):
        return {int(traj_id): np.asarray(indices, dtype=np.int64) for traj_id, indices in sensor_obj.items()}
    return np.asarray(sensor_obj, dtype=np.int64)


def _always_available_positions_from_path(always_available_path, sensor_map, entries, root):
    if sensor_map is None or entries is None:
        raise ValueError("Sensor-dropout always_available_sensor_path requires the original Chicago sensor_map and entries.")

    always_map = _load_sensor_dropout_map(always_available_path)
    if isinstance(always_map, dict):
        always_map = extend_chicago_sensor_map(entries, root, always_map)

    positions_by_traj = {}
    for traj_pos, entry in enumerate(entries):
        original_indices = np.asarray(sensor_map[entry["traj_id"]], dtype=np.int64)
        if isinstance(always_map, dict):
            requested_indices = np.asarray(always_map[entry["traj_id"]], dtype=np.int64)
        else:
            requested_indices = always_map
        keep_mask = np.isin(original_indices, requested_indices)
        positions_by_traj[traj_pos] = np.flatnonzero(keep_mask).astype(np.int64)
    return positions_by_traj


def apply_chicago_sensor_dropout(
    sensors_data,
    config,
    num_sensors,
    observable_count,
    sensor_map=None,
    entries=None,
    root=None,
):
    rate, always_available, always_available_path, seed = chicago_sensor_dropout_settings(config)
    if rate <= 0.0:
        return sensors_data, {
            "enabled": False,
            "rate": rate,
            "always_available_sensors": always_available,
            "always_available_sensor_path": always_available_path,
            "seed": seed,
            "dropped_time_steps": 0,
        }
    if rate >= 1.0:
        raise ValueError("Chicago sensor_dropout_rate must be less than 1.0 so each trajectory has an available neighbor.")

    num_sensors = int(num_sensors)
    observable_count = int(observable_count)
    sensor_feature_count = num_sensors * observable_count
    if sensor_feature_count == 0:
        return sensors_data, {
            "enabled": False,
            "rate": rate,
            "always_available_sensors": always_available,
            "always_available_sensor_path": always_available_path,
            "seed": seed,
            "dropped_time_steps": 0,
        }

    if always_available_path:
        always_positions_by_traj = _always_available_positions_from_path(
            always_available_path,
            sensor_map,
            entries,
            root,
        )
    else:
        always_positions = np.arange(min(always_available, num_sensors), dtype=np.int64)
        always_positions_by_traj = {
            traj_idx: always_positions for traj_idx in range(int(sensors_data.shape[0]))
        }

    dropped = sensors_data.clone()
    rng = np.random.default_rng(seed)
    ntraj, ntimes = int(dropped.shape[0]), int(dropped.shape[1])
    mask = rng.random((ntraj, ntimes)) < rate
    dropped_time_steps = 0
    droppable_counts = []
    always_counts = []

    all_positions = np.arange(num_sensors, dtype=np.int64)
    for traj_idx in range(ntraj):
        always_positions = np.asarray(always_positions_by_traj.get(traj_idx, np.empty(0, dtype=np.int64)), dtype=np.int64)
        always_positions = always_positions[(always_positions >= 0) & (always_positions < num_sensors)]
        droppable_sensors = np.setdiff1d(all_positions, always_positions, assume_unique=False)
        always_counts.append(int(always_positions.size))
        droppable_counts.append(int(droppable_sensors.size))
        if droppable_sensors.size == 0:
            continue

        droppable_cols = []
        for sensor_idx in droppable_sensors:
            start = int(sensor_idx) * observable_count
            droppable_cols.extend(range(start, start + observable_count))
        droppable_cols = torch.tensor(droppable_cols, dtype=torch.long, device=sensors_data.device)

        available = np.flatnonzero(~mask[traj_idx])
        if available.size == 0:
            raise ValueError(f"Sensor dropout removed every time step for trajectory index {traj_idx}.")
        missing = np.flatnonzero(mask[traj_idx])
        dropped_time_steps += int(missing.size)
        for time_idx in missing:
            nearest = int(available[np.argmin(np.abs(available - time_idx))])
            dropped[traj_idx, time_idx, droppable_cols] = dropped[traj_idx, nearest, droppable_cols]

    return dropped, {
        "enabled": True,
        "rate": rate,
        "always_available_sensors": always_available,
        "always_available_sensor_path": always_available_path,
        "always_available_counts": always_counts,
        "droppable_counts": droppable_counts,
        "seed": seed,
        "dropped_time_steps": dropped_time_steps,
    }


def _resolve_channel_indices(indices, output_size, label):
    if isinstance(indices, int):
        indices = [indices]
    indices = [int(idx) for idx in indices]
    if not indices:
        raise ValueError(f"Chicago {label}s must contain at least one channel index.")
    for idx in indices:
        if idx < 0 or idx >= output_size:
            raise ValueError(f"Chicago {label} channel {idx} is out of range for output size {output_size}.")
    return indices


def chicago_split_indices(entries, split_file=None):
    """Map the released storm-ID protocol to the sorted prepared manifest."""
    path = Path(split_file or os.environ.get("CSTRIDE_SPLIT_FILE", ROOT / "splits/chicago_split.json"))
    split = json.loads(path.read_text())
    if isinstance(entries, (int, np.integer)):
        # Compatibility for older callers that pass only the manifest length.
        ids = sorted(set(split["train_files"] + split["checkpoint_monitor_files"] + split["test"] + split["not_used"]))
        if len(ids) != entries:
            ids = sorted(split["train"] + split["test"])
            if len(ids) != entries:
                raise ValueError("Pass manifest entries when their IDs differ from the release protocol")
    else:
        ids = [int(entry["traj_id"]) for entry in entries]
    positions = {tid: i for i, tid in enumerate(ids)}
    # Resolve legacy catalog references against the published event IDs when needed.
    references = {**{85+i:25+i for i in range(5)}, **{95+i:35+i for i in range(5)},
                  **{110+i:50+i for i in range(10)}}
    def position(tid):
        event = tid if tid in positions else references.get(tid, tid)
        if event not in positions:
            raise ValueError(f"Manifest is missing split event ID: {event}")
        return positions[event]
    return tuple(np.array([position(tid) for tid in split[key]], dtype=int)
                 for key in ("train_files", "checkpoint_monitor_files", "test"))


def chicago_event_seed_indices(entries, split_file=None):
    """Retain forecast RNG indexing independently of prepared-file ordering."""
    path = Path(split_file or os.environ.get("CSTRIDE_SPLIT_FILE", ROOT / "splits/chicago_split.json"))
    split = json.loads(path.read_text())
    catalog = sorted(set(split["train_files"] + split["checkpoint_monitor_files"] + split["test"] + split["not_used"]))
    positions = {tid:i for i,tid in enumerate(catalog)}
    return np.array([positions[int(entry["traj_id"])] for entry in entries], dtype=int)


def load_sensor_map_file(path):
    path = Path(path)
    if path.suffix == ".json":
        return {int(k): np.asarray(v, dtype=np.int64) for k, v in json.loads(path.read_text()).items()}
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as values:
            return {int(k): np.asarray(values[k], dtype=np.int64) for k in values.files}
    # Compatibility with trusted legacy training maps; released maps use JSON.
    return torch.load(path, weights_only=False, map_location="cpu")


def load_chicago_normalizers(path):
    with np.load(path, allow_pickle=False) as values:
        return {k: MinMaxNormalizer(values[k + "_min"], values[k + "_max"])
                for k in ("state", "depth", "precip", "elevation", "manning", "slope")}


def fit_chicago_normalizers(entries, idx_train, root):
    cached = os.environ.get("CSTRIDE_NORMALIZERS")
    if cached:
        return load_chicago_normalizers(cached)
    state_min = None
    state_max = None
    precip_min = None
    precip_max = None
    elev_min = np.inf
    elev_max = -np.inf
    manning_min = np.inf
    manning_max = -np.inf
    slope_min = np.inf
    slope_max = -np.inf

    for idx in idx_train:
        entry = entries[idx]
        state = get_chicago_state_array(root, entry)
        precip = np.load(root / entry["precip_file"], mmap_mode="r")
        elevation = np.load(root / entry["elevation_file"], mmap_mode="r")
        manning = np.load(root / entry["manning_file"], mmap_mode="r")
        slope = np.load(root / entry["slope_file"], mmap_mode="r")

        current_state_min = np.min(state, axis=(0, 1)).astype(np.float32)
        current_state_max = np.max(state, axis=(0, 1)).astype(np.float32)
        state_min = current_state_min if state_min is None else np.minimum(state_min, current_state_min)
        state_max = current_state_max if state_max is None else np.maximum(state_max, current_state_max)
        current_precip_min = np.min(precip, axis=0)
        current_precip_max = np.max(precip, axis=0)
        precip_min = current_precip_min if precip_min is None else np.minimum(precip_min, current_precip_min)
        precip_max = current_precip_max if precip_max is None else np.maximum(precip_max, current_precip_max)
        elev_min = min(elev_min, float(np.min(elevation)))
        elev_max = max(elev_max, float(np.max(elevation)))
        manning_min = min(manning_min, float(np.min(manning)))
        manning_max = max(manning_max, float(np.max(manning)))
        slope_min = min(slope_min, float(np.min(slope)))
        slope_max = max(slope_max, float(np.max(slope)))

    return {
        "state": MinMaxNormalizer(state_min, state_max),
        "depth": MinMaxNormalizer(float(state_min[0]), float(state_max[0])),
        "precip": MinMaxNormalizer(precip_min, precip_max),
        "elevation": MinMaxNormalizer(elev_min, elev_max),
        "manning": MinMaxNormalizer(manning_min, manning_max),
        "slope": MinMaxNormalizer(slope_min, slope_max),
    }


def _pack_coordinate_rows(coords):
    coords = np.ascontiguousarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected 2D Chicago coordinates with shape (n, 2), got {coords.shape}")

    if coords.dtype == np.float16:
        packed = coords.view(np.uint16).reshape(-1, 2)
        return (packed[:, 0].astype(np.uint32) << 16) | packed[:, 1].astype(np.uint32)

    row_dtype = np.dtype((np.void, coords.dtype.itemsize * coords.shape[1]))
    return coords.view(row_dtype).reshape(-1)


def _lookup_common_indices(row_keys, common_keys):
    sort_idx = np.argsort(row_keys)
    sorted_keys = row_keys[sort_idx]
    positions = np.searchsorted(sorted_keys, common_keys)
    if positions.size != common_keys.size:
        raise ValueError("Failed to align Chicago common coordinates.")
    if np.any(positions >= sorted_keys.shape[0]) or not np.array_equal(sorted_keys[positions], common_keys):
        raise ValueError("Chicago common-coordinate intersection contains keys missing from a trajectory.")
    return sort_idx[positions]


def get_chicago_common_coordinate_map(entries, root):
    root = Path(root)
    reference_coords = np.load(root / entries[0]["coords_file"], mmap_mode="r")
    common_keys = np.unique(_pack_coordinate_rows(reference_coords))

    for entry in entries[1:]:
        coords = np.load(root / entry["coords_file"], mmap_mode="r")
        keys = np.unique(_pack_coordinate_rows(coords))
        common_keys = np.intersect1d(common_keys, keys, assume_unique=True)
        if common_keys.size == 0:
            raise ValueError("No common Chicago coordinates remain across all trajectories.")

    common_index_map = {}
    common_coords = None
    for entry in entries:
        coords = np.load(root / entry["coords_file"], mmap_mode="r")
        row_keys = _pack_coordinate_rows(coords)
        indices = _lookup_common_indices(row_keys, common_keys)
        common_index_map[entry["traj_id"]] = indices.astype(np.int64)
        if common_coords is None:
            common_coords = np.asarray(coords[indices], dtype=np.float32)

    return common_coords, common_index_map
def build_chicago_qr_sensor_map(entries, root, config):
    import pysensors

    common_coords, common_index_map = get_chicago_common_coordinate_map(entries, root)
    ntimes = int(get_chicago_state_array(root, entries[0]).shape[0])
    total_snapshots = len(entries) * ntimes
    qr_num_snapshots = config["basics"].get("qr_num_snapshots")
    if qr_num_snapshots is None:
        selected_global = np.arange(total_snapshots, dtype=np.int64)
    else:
        qr_num_snapshots = int(qr_num_snapshots)
        if qr_num_snapshots <= 0:
            raise ValueError("Chicago qr_num_snapshots must be positive when provided.")
        if qr_num_snapshots > total_snapshots:
            raise ValueError(
                f"Chicago qr_num_snapshots={qr_num_snapshots} exceeds total snapshots={total_snapshots}."
            )
        rng = np.random.default_rng(int(config["basics"]["random_seed"]))
        selected_global = np.sort(rng.choice(total_snapshots, size=qr_num_snapshots, replace=False))

    n_snapshots = selected_global.shape[0]
    n_common = common_coords.shape[0]
    if n_common < config["basics"]["num_sensors"]:
        raise ValueError(
            f"Chicago QR placement only has {n_common} common cells across trajectories, "
            f"which is fewer than num_sensors={config['basics']['num_sensors']}."
        )

    row_specs_by_traj = {}
    for row_idx, global_idx in enumerate(selected_global.tolist()):
        traj_pos = global_idx // ntimes
        time_idx = global_idx % ntimes
        row_specs_by_traj.setdefault(traj_pos, []).append((row_idx, time_idx))

    with tempfile.TemporaryDirectory(prefix="chicago_qr_", dir="/tmp") as tmpdir:
        matrix_path = Path(tmpdir) / "snapshot_matrix.dat"
        snapshot_matrix = np.memmap(matrix_path, mode="w+", dtype=np.float32, shape=(n_snapshots, n_common))

        for traj_pos, row_specs in row_specs_by_traj.items():
            entry = entries[traj_pos]
            common_indices = common_index_map[entry["traj_id"]]
            depth = get_chicago_state_array(root, entry)[:, common_indices, 0]
            row_indices = [row_idx for row_idx, _ in row_specs]
            time_indices = [time_idx for _, time_idx in row_specs]
            block = np.asarray(depth[time_indices], dtype=np.float32)
            snapshot_matrix[row_indices] = block

        snapshot_matrix.flush()
        sspor = pysensors.reconstruction.SSPOR(
            basis=pysensors.basis.Identity(n_basis_modes=n_snapshots),
            n_sensors=config["basics"]["num_sensors"],
        )
        sspor.fit(snapshot_matrix)
        sensor_common_indices = np.asarray(sspor.get_selected_sensors(), dtype=np.int64)

    sensor_map = {}
    for entry in entries:
        sensor_map[entry["traj_id"]] = common_index_map[entry["traj_id"]][sensor_common_indices]
    return sensor_map


def build_sensor_anchors(config, reference_coords, reference_depth=None):
    num_sensors = config["basics"]["num_sensors"]
    placement = config["basics"]["sensor_placement"]
    if placement == "random":
        chosen = np.random.choice(reference_coords.shape[0], size=num_sensors, replace=False)
        return reference_coords[chosen]

    if placement == "uniform":
        sx = max(1, int(np.sqrt(num_sensors)))
        sy = max(1, math.ceil(num_sensors / sx))
        xs = np.linspace(reference_coords[:, 0].min(), reference_coords[:, 0].max(), sx, dtype=np.float32)
        ys = np.linspace(reference_coords[:, 1].min(), reference_coords[:, 1].max(), sy, dtype=np.float32)
        anchors = np.array([[x, y] for x in xs for y in ys], dtype=np.float32)
        return anchors[:num_sensors]

    if placement == "QR":
        if reference_depth is None:
            raise ValueError("QR sensor placement requires reference depth data.")
        import pysensors

        SSPOR = pysensors.reconstruction.SSPOR(
            basis=pysensors.basis.Identity(n_basis_modes=96),
            n_sensors=num_sensors,
        )
        SSPOR.fit(reference_depth[..., 0].reshape(-1, reference_coords.shape[0]))
        indices = SSPOR.get_selected_sensors()
        return reference_coords[indices]

    raise ValueError(f"Unsupported sensor placement for Chicago dataset: {placement}")


def map_anchors_to_coords(anchors, coords):
    chosen = []
    used = np.zeros(coords.shape[0], dtype=bool)
    for anchor in anchors:
        distances = np.sum((coords - anchor[None, :]) ** 2, axis=1)
        distances[used] = np.inf
        idx = int(np.argmin(distances))
        chosen.append(idx)
        used[idx] = True
    return np.asarray(chosen, dtype=np.int64)


def save_chicago_sensors(save_path, run_name, sensor_map):
    target_dir = Path(save_path) / run_name
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "sensor_indices.json").write_text(json.dumps({str(k): np.asarray(v).tolist() for k, v in sensor_map.items()}, indent=2) + "\n")


def load_chicago_sensors(path, config):
    num_sensors = int(config["basics"].get("num_sensors", 0))
    if num_sensors == 0:
        return {
            config_entry["traj_id"]: np.empty(0, dtype=np.int64)
            for config_entry in load_chicago_manifest(config["basics"].get("data_root", "data/Chicago"))[2]
        }

    saved_sensor_idx_path = Path(path) / "sensor_indices.json"
    if not saved_sensor_idx_path.exists():
        saved_sensor_idx_path = Path(path) / "sensor_indices.pt"
    if saved_sensor_idx_path.exists():
        sensor_map = load_sensor_map_file(saved_sensor_idx_path)
        if isinstance(sensor_map, dict):
            return sensor_map
        raise ValueError(f"Chicago sensor file must contain a trajectory-index map, got {type(sensor_map)}")
    sensor_idx_path = config["basics"].get("sensor_idx_path")
    if sensor_idx_path:
        sensor_map = load_sensor_map_file(sensor_idx_path)
        if isinstance(sensor_map, dict):
            return sensor_map
        raise ValueError(f"Chicago sensor file must contain a trajectory-index map, got {type(sensor_map)}")
    return None


def extend_chicago_sensor_map(entries, root, sensor_map):
    if sensor_map is None:
        return None

    extended_map = {int(traj_id): np.asarray(indices, dtype=np.int64) for traj_id, indices in sensor_map.items()}
    missing_entries = [entry for entry in entries if entry["traj_id"] not in extended_map]
    if not missing_entries:
        return extended_map

    nonempty_reference = None
    for entry in entries:
        traj_id = entry["traj_id"]
        indices = extended_map.get(traj_id)
        if indices is not None and len(indices) > 0:
            nonempty_reference = (entry, indices)
            break

    if nonempty_reference is None:
        for entry in missing_entries:
            extended_map[entry["traj_id"]] = np.empty(0, dtype=np.int64)
        return extended_map

    reference_entry, reference_indices = nonempty_reference
    reference_coords = np.load(Path(root) / reference_entry["coords_file"], mmap_mode="r")
    anchors = np.asarray(reference_coords[reference_indices], dtype=np.float32)

    for entry in missing_entries:
        coords = np.load(Path(root) / entry["coords_file"], mmap_mode="r")
        extended_map[entry["traj_id"]] = map_anchors_to_coords(anchors, coords)

    return extended_map


def build_chicago_sensor_map(entries, idx_train, root, config):
    num_sensors = int(config["basics"].get("num_sensors", 0))
    if num_sensors == 0:
        return {entry["traj_id"]: np.empty(0, dtype=np.int64) for entry in entries}

    if config["basics"]["sensor_placement"] == "QR":
        return build_chicago_qr_sensor_map(entries, root, config)

    reference_entry = entries[int(idx_train[0])]
    reference_coords = np.load(root / reference_entry["coords_file"], mmap_mode="r")
    reference_depth = get_chicago_state_array(root, reference_entry)[..., :1]
    anchors = build_sensor_anchors(config, reference_coords, reference_depth)
    sensor_map = {}
    for entry in entries:
        coords = np.load(root / entry["coords_file"], mmap_mode="r")
        sensor_map[entry["traj_id"]] = map_anchors_to_coords(anchors, coords)
    return sensor_map


def build_sensor_sequences(entries, root, sensor_map, normalizers, observable_indices, include_precipitation=True):
    sequences = []
    for entry in entries:
        traj_id = entry["traj_id"]
        state = get_chicago_state_array(root, entry)
        sensor_idx = sensor_map[traj_id]
        if len(sensor_idx) == 0:
            state_sensor = np.empty((state.shape[0], 0), dtype=np.float32)
        else:
            state_sensor = normalizers["state"].transform(state[:, sensor_idx, :])
            state_sensor = state_sensor[:, :, observable_indices].reshape(state.shape[0], -1)
        features = [state_sensor]
        if include_precipitation:
            precip = np.load(root / entry["precip_file"], mmap_mode="r")
            features.append(normalizers["precip"].transform(precip))
        sequence = np.concatenate(features, axis=-1) if len(features) > 1 else features[0]
        sequences.append(torch.tensor(sequence, dtype=torch.float32))
    return sequences


def make_windows(sensors_data, lag):
    if sensors_data.ndim == 2:
        ntimes, _ = sensors_data.shape
        nt_valid = ntimes - lag
        return torch.stack([sensors_data[k : k + lag] for k in range(nt_valid)], dim=0)

    n_traj, ntimes, _ = sensors_data.shape
    nt_valid = ntimes - lag
    windows = torch.stack([sensors_data[:, k : k + lag, :] for k in range(nt_valid)], dim=1)
    return windows.reshape(n_traj * nt_valid, lag, -1)


def compute_chicago_snapshot_weights(entries, subset_indices, root, lag):
    weights = []
    for idx in subset_indices:
        depth = get_chicago_state_array(root, entries[idx])[..., :1]
        depth_slice = np.asarray(depth[lag:, :, 0], dtype=np.float32)
        norms = np.linalg.norm(depth_slice, axis=1).astype(np.float32)
        if not np.isfinite(norms).all():
            bad = np.count_nonzero(~np.isfinite(norms))
            raise ValueError(
                f"Encountered {bad} non-finite snapshot norms while building Chicago importance weights "
                f"for traj_id={entries[idx]['traj_id']}."
            )
        floor = max(float(norms.mean()) * 1e-3, 1e-6)
        weights.extend((norms + floor).tolist())
    weights = torch.tensor(weights, dtype=torch.float32)
    if not torch.isfinite(weights).all():
        bad = int((~torch.isfinite(weights)).sum())
        raise ValueError(f"Chicago snapshot weights contain {bad} non-finite values after construction.")
    return weights


def compute_chicago_spatial_probs(entries, subset_indices, root, lag):
    probs = {}
    for idx in subset_indices:
        entry = entries[idx]
        depth = get_chicago_state_array(root, entry)[..., :1]
        depth_slice = np.asarray(depth[lag:, :, 0], dtype=np.float32)
        watermark = np.max(np.abs(depth_slice), axis=0).astype(np.float64)
        if not np.isfinite(watermark).all():
            bad = np.count_nonzero(~np.isfinite(watermark))
            raise ValueError(
                f"Encountered {bad} non-finite spatial watermark values while building Chicago sampling "
                f"probabilities for traj_id={entry['traj_id']}."
            )
        floor = max(float(watermark.mean()) * 1e-3, 1e-12)
        watermark = watermark + floor
        total = watermark.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError(
                f"Invalid Chicago spatial probability normalizer for traj_id={entry['traj_id']}: {total}"
            )
        probs[entry["traj_id"]] = watermark / total
    return probs


class ChicagoTrajectoryDataset(Dataset):
    def __init__(
        self,
        entries,
        root,
        input_windows,
        nt,
        lag,
        normalizers,
        target_indices,
        terrain_conditioning=True,
        num_points=None,
        spatial_probs=None,
    ):
        super().__init__()
        self.entries = entries
        self.root = Path(root)
        self.x_data = input_windows.float().cpu()
        self.nt = int(nt)
        self.lag = int(lag)
        self.normalizers = normalizers
        self.target_indices = list(target_indices)
        self.target_normalizer = get_chicago_target_normalizer(normalizers, self.target_indices)
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
            raise ValueError(
                f"x_data samples ({self.x_data.shape[0]}) do not match trajectories ({self.num_trajectories}) * nt ({self.nt})"
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        traj_idx = idx // self.nt
        time_idx = idx % self.nt
        entry = self.entries[traj_idx]

        state_t = self.state_arrays[traj_idx][self.lag + time_idx]
        coords = self.coords_arrays[traj_idx]
        if self.terrain_conditioning:
            elevation = self.elev_arrays[traj_idx]
            manning = self.manning_arrays[traj_idx]
            slope = self.slope_arrays[traj_idx]

        if self.num_points is not None:
            nstate = coords.shape[0]
            sample_count = min(self.num_points, nstate)
            probs = self.spatial_probs.get(entry["traj_id"])
            indices = np.random.choice(nstate, size=sample_count, replace=False, p=probs)
            state_t = state_t[indices]
            coords = coords[indices]
            if self.terrain_conditioning:
                elevation = elevation[indices]
                manning = manning[indices]
                slope = slope[indices]

        target = state_t[:, self.target_indices]
        target = self.target_normalizer.transform(target).astype(np.float32)
        coords = coords.astype(np.float32)
        if self.terrain_conditioning:
            elevation = self.normalizers["elevation"].transform(elevation)[:, None]
            manning = self.normalizers["manning"].transform(manning)[:, None]
            slope = self.normalizers["slope"].transform(slope)[:, None]
            coords = np.concatenate([coords, elevation, manning, slope], axis=-1)

        return (
            self.x_data[idx],
            torch.tensor(coords, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
        )
