import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import argparse
import os
import ast
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.ticker import MaxNLocator

from plot_chicago_future_precipitation import select_section
from cstride.chicago import MinMaxNormalizer, chicago_split_indices, load_chicago_manifest, chicago_event_seed_indices


DEFAULT_RUN = ROOT / Path("checkpoints/full")
DEFAULT_OUTPUT = ROOT / Path("figures/fig08_rainfall_mismatch.png")
START_PREFIX = "Evaluation start steps for trajectory index "


def make_precip_mismatch_seed(base_seed, entry_idx, start_step):
    return (int(base_seed) + 1_000_003 * int(entry_idx) + 9_176 * int(start_step)) % (2**63 - 1)


def impose_precip_mismatch(future_precipitation, precip_mismatch, seed):
    generator = torch.Generator(device=future_precipitation.device)
    generator.manual_seed(int(seed) % (2**63 - 1))
    noise = torch.randn(
        future_precipitation.shape,
        dtype=future_precipitation.dtype,
        device=future_precipitation.device,
        generator=generator,
    )
    horizon = future_precipitation.shape[1]
    growth = torch.arange(1, horizon + 1, dtype=future_precipitation.dtype)
    growth = growth.view(1, horizon, 1) / float(horizon)
    step_std = torch.std(future_precipitation, dim=-1, keepdim=True, unbiased=False)
    perturbed = future_precipitation + float(precip_mismatch) * growth * step_std * noise
    return torch.clamp(perturbed, -1.0, 1.0)


def fit_precipitation_normalizer(entries, root):
    idx_train, _, _ = chicago_split_indices(entries)
    precip_min = None
    precip_max = None
    for entry_idx in idx_train:
        precip = np.load(root / entries[int(entry_idx)]["precip_file"], mmap_mode="r")
        current_min = np.min(precip, axis=0)
        current_max = np.max(precip, axis=0)
        precip_min = current_min if precip_min is None else np.minimum(precip_min, current_min)
        precip_max = current_max if precip_max is None else np.maximum(precip_max, current_max)
    return MinMaxNormalizer(precip_min, precip_max)


def parse_evaluation_starts(section, entries=None):
    starts = {}
    positions = None if entries is None else {int(seed):i for i,seed in enumerate(chicago_event_seed_indices(entries))}
    for key, value in section["metadata"].items():
        if key.startswith(START_PREFIX):
            entry_idx = int(key.removeprefix(START_PREFIX))
            if positions is not None:
                entry_idx = positions[entry_idx]
            starts[entry_idx] = [int(step) for step in ast.literal_eval(value)]
    if not starts:
        raise ValueError("The selected results section does not document evaluation start steps.")
    return starts


def compute_step_intensities(run_dir, mismatch):
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    root, _, entries = load_chicago_manifest(config["basics"].get("data_root", "data/Chicago"))
    section = select_section(run_dir, True, mismatch, True)
    metadata = section["metadata"]
    horizon = int(metadata["Autoregressive horizon"])
    eval_seed = int(metadata["Evaluation seed"])
    starts_by_entry = parse_evaluation_starts(section, entries)
    seed_indices = chicago_event_seed_indices(entries)
    normalizer = fit_precipitation_normalizer(entries, root)

    precipitation_samples = []
    mismatch_samples = []
    signed_samples = []
    for entry_idx, start_steps in starts_by_entry.items():
        precip = np.load(root / entries[entry_idx]["precip_file"], mmap_mode="r")
        for start_step in start_steps:
            true_precip = np.asarray(precip[start_step : start_step + horizon], dtype=np.float32)
            if true_precip.shape[0] != horizon:
                raise ValueError(
                    f"Trajectory index {entry_idx}, start {start_step} has only "
                    f"{true_precip.shape[0]} of {horizon} requested steps."
                )
            normalized = normalizer.transform(true_precip)
            seed = make_precip_mismatch_seed(eval_seed, seed_indices[entry_idx], start_step)
            noisy_normalized = impose_precip_mismatch(
                torch.tensor(normalized[None], dtype=torch.float32), mismatch, seed
            )[0].numpy()
            noisy_precip = normalizer.inverse_transform(noisy_normalized)

            precipitation_samples.append(np.mean(true_precip, axis=1))
            mismatch_samples.append(np.mean(np.abs(noisy_precip - true_precip), axis=1))
            signed_samples.append(np.mean(noisy_precip - true_precip, axis=1))

    precipitation_samples = np.asarray(precipitation_samples)
    mismatch_samples = np.asarray(mismatch_samples)
    return {
        "precipitation_mean": precipitation_samples.mean(axis=0),
        "precipitation_std": precipitation_samples.std(axis=0),
        "mismatch_mean": mismatch_samples.mean(axis=0),
        "mismatch_std": mismatch_samples.std(axis=0),
        "sample_count": precipitation_samples.shape[0],
        "signed_mismatch_mean": np.asarray(signed_samples).mean(axis=0),
        "cumulative_signed_mismatch_per_rollout": np.asarray(signed_samples).sum(axis=1),
    }


def plot_intensities(stats, mismatch, output_path):
    steps = np.arange(1, len(stats["precipitation_mean"]) + 1)
    fig, ax = plt.subplots(figsize=(6, 3))
    series = (
        ("True precipitation", stats["precipitation_mean"], stats["precipitation_std"], "#2a7f62", "-"),
        ("Mean absolute mismatch", stats["mismatch_mean"], stats["mismatch_std"], "#d97706", "--"),
    )
    for label, mean, std, color, linestyle in series:
        ax.plot(
            steps,
            mean,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            marker="o",
            markersize=3,
            label=label,
        )
        if label == "Mean absolute mismatch":
            ax.fill_between(
                steps,
                mean - std,
                mean + std,
                color=color,
                alpha=0.12,
                linewidth=0,
            )

    ax.set_xlim(1, steps[-1])
    ax.set_ylim(0, 1.5)
    ax.set_xlabel("Forecast step")
    ax.set_ylabel("Mean precipitation intensity (mm/h)")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare precipitation and imposed mismatch intensity over a documented rollout."
    )
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--mismatch", type=float, default=1.0)
    parser.add_argument("--stats-output", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root",type=Path,help="Prepared Chicago arrays")
    args = parser.parse_args()
    if args.data_root is not None: os.environ["CSTRIDE_DATA_ROOT"]=str(args.data_root)
    if args.mismatch <= 0.0:
        raise ValueError(f"mismatch must be positive, got {args.mismatch}.")

    if args.data_root is None:
        if not np.isclose(args.mismatch, 1.0):
            parser.error("Custom mismatch requires --data-root")
        stats = json.loads((ROOT / "results/precipitation_mismatch.json").read_text())
        stats = {k: np.asarray(v) if isinstance(v, list) else v for k,v in stats.items()}
    else:
        stats = compute_step_intensities(args.run, args.mismatch)
    if args.stats_output is not None:
        args.stats_output.write_text(json.dumps({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in stats.items()}, indent=2))
    plot_intensities(stats, args.mismatch, args.output)
    print(f"Samples: {stats['sample_count']}")
    print(f"Mean precipitation intensity: {np.mean(stats['precipitation_mean']):.6f}")
    print(f"Mean absolute mismatch intensity: {np.mean(stats['mismatch_mean']):.6f}")
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
