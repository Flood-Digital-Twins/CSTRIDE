"""Portable runtime overrides and checkpoint wrapper compatibility."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import torch
import yaml


def parse_args(parser):
    parser.add_argument("--data-root", help="Directory containing the Chicago manifest.json")
    parser.add_argument("--device", help="PyTorch device, e.g. cpu or cuda:0")
    parser.add_argument("--num-workers", type=int, help="DataLoader workers (0 for debugging)")
    parser.add_argument("--normalizers", help="Cached normalization .npz; omit to fit on training arrays")
    parser.add_argument("--split-file", help="Storm-ID split JSON")
    args = parser.parse_args()
    for name in ("data_root", "device", "num_workers", "normalizers", "split_file"):
        value = getattr(args, name)
        if value is not None:
            os.environ["CSTRIDE_" + name.upper()] = str(value)
    return args


def load_config(stream):
    config = yaml.safe_load(stream)
    for section, names in (("basics", ("data_root", "sensor_idx_path", "sensor_dropout_always_available_sensor_path", "split_file")), ("training", ("save_path",))):
        for name in names:
            value = config.get(section, {}).get(name)
            if value and not Path(value).is_absolute():
                config[section][name] = str(ROOT / value)
    for name in ("data_root", "device", "split_file"):
        value = os.environ.get("CSTRIDE_" + name.upper())
        if value is not None:
            config["basics"][name] = value
    workers = os.environ.get("CSTRIDE_NUM_WORKERS")
    if workers is not None:
        if int(workers) < 0:
            raise ValueError("num-workers must be nonnegative")
        config.setdefault("training", {})["num_workers"] = int(workers)
    return config


def load_state_dict(path, device="cpu"):
    state = torch.load(path, map_location=device, weights_only=True)
    cleaned = {}
    for key, value in state.items():
        while key.startswith(("module.", "_orig_mod.")):
            key = key.split(".", 1)[1]
        cleaned[key] = value
    return cleaned
