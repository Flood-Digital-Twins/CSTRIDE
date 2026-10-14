import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import argparse
import re
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_PRECIP_RUN = ROOT / Path("checkpoints/full")
DEFAULT_NO_PRECIP_RUN = ROOT / Path("checkpoints/terrain")
DEFAULT_OUTPUT = ROOT / Path("figures/fig09_forecast_errors.png")

SECTION_PREFIX = "Chicago Reconstruction-Backbone Observation-Prediction Evaluation"
ERROR_PATTERN = re.compile(
    r"^Forecast step (\d+) denormalized reconstruction relative error with predicted observations: "
    r"mean=([^ ]+) \([^)]*\), std=([^ ]+)$"
)


def select_section(run_dir, include_precipitation, precip_mismatch, obs_include_precipitation):
    selected = json.loads((ROOT / "results/forecasts.json").read_text())
    candidates = []
    for label, record in selected.items():
        model = "full" if label == "noisy" else label
        if model != Path(run_dir).name:
            continue
        section = record["section"]
        metadata = section["metadata"]
        if (metadata.get("include_precipitation_input") == include_precipitation
            and metadata.get("Observation LSTM include_precipitation_input") == obs_include_precipitation
            and np.isclose(float(metadata.get("Precipitation mismatch", 0.0)), precip_mismatch)):
            section["means"] = {int(k): v for k, v in section["means"].items()}
            section["stds"] = {int(k): v for k, v in section["stds"].items()}
            candidates.append(section)
    if len(candidates) != 1:
        raise ValueError("No unique matching paper forecast scenario")
    return candidates[0]


def evaluation_signature(section):
    metadata = section["metadata"]
    signature_keys = (
        "Split",
        "Observed time fraction",
        "Observed training steps",
        "Autoregressive horizon",
        "Evaluation seed",
        "Random valid starts per trajectory",
    )
    signature = {key: metadata.get(key) for key in signature_keys}
    signature["start_steps"] = {
        key: value
        for key, value in metadata.items()
        if key.startswith("Evaluation start steps for trajectory index ")
    }
    return signature


def validate_comparable(scenarios):
    reference_label, reference = scenarios[0]
    reference_signature = evaluation_signature(reference)
    reference_steps = sorted(reference["means"])

    for label, section in scenarios[1:]:
        if evaluation_signature(section) != reference_signature:
            raise ValueError(f"{label} does not use the same evaluation setting as {reference_label}.")
        if sorted(section["means"]) != reference_steps:
            raise ValueError(f"{label} does not report the same forecast steps as {reference_label}.")


def plot_scenarios(scenarios, output_path):
    styles = {
        "True": {"color": "#2a7f62", "linestyle": "-"},
        "Noisy": {"color": "#d97706", "linestyle": "--"},
        "Terrain-only": {"color": "#52606d", "linestyle": "-."},
        "Vanilla": {"color": "#9467bd", "linestyle": ":"},
        "Persistence": {"color": "black", "linestyle": ":"},
    }
    fig, ax = plt.subplots(figsize=(6, 3))

    for label, section in scenarios:
        steps = np.asarray(sorted(section["means"]))
        means = np.asarray([section["means"][step] for step in steps])
        stds = np.asarray([section["stds"][step] for step in steps])
        style = styles[label]
        ax.plot(
            steps,
            means,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            marker="o",
            markersize=3,
            label=label,
        )
        ax.fill_between(
            steps,
            means - stds,
            means + stds,
            color=style["color"],
            alpha=0.12,
            linewidth=0,
        )

    ax.set_xlim(1, max(max(section["means"]) for _, section in scenarios))
    ax.set_ylim(0, 0.45)
    ax.set_xlabel("Forecast step")
    ax.set_ylabel("Mean relative $L_2$ depth error")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Overlay Chicago autoregressive forecast error for future-precipitation scenarios."
    )
    parser.add_argument("--precip-run", type=Path, default=DEFAULT_PRECIP_RUN)
    parser.add_argument("--no-precip-run", type=Path, default=DEFAULT_NO_PRECIP_RUN)
    parser.add_argument("--vanilla-run", type=Path, default=ROOT / Path("checkpoints/vanilla"))
    parser.add_argument("--persistence", type=Path, default=None)
    parser.add_argument("--mismatch", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    scenarios = [
        (
            "True",
            select_section(args.precip_run, True, 0.0, True),
        ),
        (
            "Noisy",
            select_section(args.precip_run, True, args.mismatch, True),
        ),
        (
            "Terrain-only",
            select_section(args.no_precip_run, False, 0.0, False),
        ),
    ]
    scenarios.append(("Vanilla", select_section(args.vanilla_run, False, 0.0, False)))
    if args.persistence is not None:
        persistence = json.loads(args.persistence.read_text())
        for key in ("means", "stds"):
            persistence[key] = {int(k): v for k, v in persistence[key].items()}
        scenarios.append(("Persistence", persistence))
    validate_comparable(scenarios)
    plot_scenarios(scenarios, args.output)

    for label, section in scenarios:
        last_step = max(section["means"])
        print(f"{label}: step {last_step} mean={section['means'][last_step]:.6f}")
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
