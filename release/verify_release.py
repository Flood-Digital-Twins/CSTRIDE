#!/usr/bin/env python3
"""Consistency checks for the C-STRIDE reproducibility package.

Run from anywhere; paths resolve from this file's location. Standard library only.

    python release/verify_release.py              # structure and content checks (seconds)
    python release/verify_release.py --checksums  # also re-hash every file in checkpoints/manifest.json
    python release/verify_release.py --root DIR   # check another tree with the same layout

Each check prints PASS / FAIL / WARN; the exit code is the number of FAILs.
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import re
import sys

# Facts the paper states and the package must agree with (arXiv:2609.39005v1).
TEST_IDS = [107, 108, 109, 120]          # four test events, including the April 2013 flood
N_TRAIN = 90                              # training events
# Four conditioning variants (Tables 5 to 9), four window lengths (Fig. 10), three sensor
# configurations and three noise levels (Table 10) besides the default.
N_MODELS = 13
STAGING_ONLY = {"CHECKLIST.md", "DEPOSITS.md"}      # root files that are not exported
NOT_EXPORTED = ("outputs", ".pytest_cache", "private")        # folders that are not exported
GITHUB_LIMIT = 95 * 2 ** 20
ABSOLUTE = re.compile(r"(?<![\w.])/(?:work2?|storage|cedar|nethome|home)/[\w.-]+")
TEXT = (".py", ".md", ".yaml", ".yml", ".json", ".txt", ".toml", ".sh", ".cfg", ".csv")

fails = 0


def report(status, name, detail=""):
    global fails
    if status == "FAIL":
        fails += 1
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))


def need(root, name, paths, what=""):
    missing = [p for p in paths if not glob.glob(os.path.join(root, p))]
    report("FAIL" if missing else "PASS", name, ("missing " + ", ".join(missing)) if missing else what)


def exported(root):
    """Files that would go to GitHub, as paths relative to the root."""
    for d, dirs, files in os.walk(root):
        rel = os.path.relpath(d, root)
        dirs[:] = [x for x in dirs if x != ".git" and os.path.normpath(os.path.join(rel, x)) not in NOT_EXPORTED]
        for f in files:
            if rel == "." and f in STAGING_ONLY:
                continue
            yield os.path.normpath(os.path.join(rel, f))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_code(root):
    need(root, "code: package", [f"code/cstride/{m}.py" for m in
                                 ("__init__", "models", "chicago", "train_next_step", "evaluate_next_step",
                                  "forecast_observations")])
    need(root, "code: launchers", [f"scripts/{m}.py" for m in ("train_next_step", "evaluate_next_step",
                                                               "forecast_observations")])
    need(root, "code: baseline and timing scripts", ["scripts/score_cldnet_baselines.py", "scripts/time_inference.py"])
    need(root, "code: tests and packaging", ["tests/test_*.py", "pyproject.toml", "requirements.txt"])
    need(root, "code: preprocessing of SynxFlow output", ["preprocessing/prepare_arrays/*.py"])
    need(root, "docs", ["README.md", "data/README.md", "preprocessing/prepare_arrays/README.md"])
    need(root, "license", ["LICENSE*"])


def check_models(root, checksums):
    ck = os.path.join(root, "checkpoints")
    models = sorted(d for d in glob.glob(os.path.join(ck, "*")) if os.path.isdir(d))
    incomplete = [os.path.basename(d) for d in models
                  if not (glob.glob(os.path.join(d, "*_model.ckpt")) and os.path.exists(os.path.join(d, "config.yaml")))]
    report("PASS" if len(models) >= N_MODELS and not incomplete else "FAIL", "checkpoints: model folders",
           f"{len(models)} of {N_MODELS}" + (f"; without weights or config.yaml: {incomplete}" if incomplete else ""))
    need(root, "checkpoints: README", ["checkpoints/README.md"])
    configs = glob.glob(os.path.join(root, "configs", "next_step", "*.y*ml"))
    report("PASS" if len(configs) >= N_MODELS else "FAIL", "configs: one per model", f"{len(configs)} of {N_MODELS}")
    mf = os.path.join(ck, "manifest.json")
    if not os.path.exists(mf):
        return report("FAIL", "checkpoints: manifest.json", "missing")
    manifest = json.load(open(mf))
    bad = []
    for rel, rec in manifest.items():
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            bad.append(f"{rel} missing")
        elif os.path.getsize(p) != rec["bytes"]:
            bad.append(f"{rel} size")
        elif checksums and sha256(p) != rec["sha256"]:
            bad.append(f"{rel} sha256")
    report("FAIL" if bad else "PASS", "checkpoints: files match manifest.json" + ("" if checksums else " (sizes)"),
           "; ".join(bad[:5]) if bad else f"{len(manifest)} files")
    listed = {os.path.normpath(k) for k in manifest}
    extra = [os.path.relpath(p, root) for p in glob.glob(os.path.join(ck, "*", "*"))
             if os.path.normpath(os.path.relpath(p, root)) not in listed]
    report("FAIL" if extra else "PASS", "checkpoints: every file is in manifest.json", ", ".join(extra[:5]))


def check_split(root):
    p = os.path.join(root, "splits", "chicago_split.json")
    if not os.path.exists(p):
        return report("FAIL", "split", "splits/chicago_split.json missing (a draft is in splits/)")
    s = json.load(open(p))
    train, test = s.get("train", []), s.get("test", [])
    problems = []
    if sorted(test) != TEST_IDS:
        problems.append(f"test is {test}, the paper's is {TEST_IDS}")
    if len(set(train)) != N_TRAIN or len(train) != N_TRAIN:
        problems.append(f"{len(set(train))} training events, the paper says {N_TRAIN}")
    if s.get("total_events") != 94 or len(set(train + test)) != 94:
        problems.append("expected 94 events in total: 90 training and 4 test")
    if set(train) & set(test):
        problems.append("train and test overlap")
    if not s.get("checkpoint_monitor_files"):
        problems.append("checkpoint monitoring schedule missing")
    if "_to_confirm" in s or "_status" in s:
        problems.append("still carries the draft's _status or _to_confirm")
    report("FAIL" if problems else "PASS", "split", "; ".join(problems))


def check_data(root):
    mapping = glob.glob(os.path.join(root, "data/gauges/*.csv")) + glob.glob(os.path.join(root, "data/gauges/*.json"))
    report("PASS" if mapping else "FAIL", "data: gauge mapping", "" if mapping else "no CSV or JSON in data/gauges/")
    need(root, "data: simulated gauge depths", ["data/gauges/simulated_depth.npz"])
    need(root, "data: USGS records of April 2013", ["data/gauges/usgs_2013/*"])
    need(root, "data: normalization statistics", ["data/normalization/*"])
    need(root, "data: evaluation grids of the test storms", [f"data/grids/grid_traj{k}.npz" for k in TEST_IDS])
    need(root, "data: compact-input checksums", ["data/SHA256SUMS"])
    need(root, "results", ["results/README.md", "results/*.json", "results/*.csv"])
    p = os.path.join(root, "data", "storms", "storm_periods.csv")
    if not os.path.exists(p):
        return report("FAIL", "data: storm periods", "data/storms/storm_periods.csv missing")
    rows = list(csv.DictReader(open(p)))
    on = [int(r["traj_id"]) for r in rows if r["on_basin"] == "yes"]
    ok = len(rows) == 94 and on == [120] and all(r["start_utc"] and r["end_utc"] for r in rows)
    report("PASS" if ok else "FAIL", "data: storm periods", f"{len(rows)} events, on the basin: {on}")


def check_hygiene(root):
    if os.path.basename(os.path.normpath(root)) != "cstride":
        internal = [name for name in (*STAGING_ONLY, "private") if os.path.exists(os.path.join(root, name))]
        report("FAIL" if internal else "PASS", "no internal staging records in publication tree", ", ".join(internal))
    files = list(exported(root))
    fields = [f for f in files if re.search(r"(state|depth|flow_variables)_traj\d+\.npy$", f)]
    report("FAIL" if fields else "PASS", "no simulated flow fields in the export", ", ".join(fields[:5]))
    sizes = {f: os.path.getsize(os.path.join(root, f)) for f in files}
    big = [f"{f} ({s / 2 ** 20:.0f} MB)" for f, s in sizes.items() if s > GITHUB_LIMIT]
    report("FAIL" if big else "PASS", "no file over GitHub's limit", ", ".join(big[:5]))
    total = sum(sizes.values()) / 2 ** 20
    report("WARN" if total > 500 else "PASS", "export size", f"{total:.0f} MB in {len(files)} files")
    hits = []
    for f in files:
        if f.endswith(TEXT) and sizes[f] < 2 ** 20 and not f.startswith("release" + os.sep):
            with open(os.path.join(root, f), errors="replace") as fh:
                for n, line in enumerate(fh, 1):
                    if ABSOLUTE.search(line):
                        hits.append(f"{f}:{n}")
                        break
    report("FAIL" if hits else "PASS", "no absolute paths in exported text files", ", ".join(hits[:8]))
    entity = [f for f in files if (f.startswith("configs") or f.startswith("checkpoints")) and f.endswith((".yaml", ".yml"))
              and "entity" in open(os.path.join(root, f), errors="replace").read()]
    report("FAIL" if entity else "PASS", "no configured tracking account", ", ".join(entity[:5]))
    pickles = [f for f in files if f.endswith(".pt")]
    report("WARN" if pickles else "PASS", "sensor maps are not pickles",
           (f"{len(pickles)} .pt files, e.g. {pickles[0]}; convert to JSON or .npz") if pickles else "")
    junk = [f for f in files if "__pycache__" in f or f.endswith(".pyc") or ".ipynb_checkpoints" in f or f.startswith("wandb")]
    report("WARN" if junk else "PASS", "no caches or logs", ", ".join(junk[:5]))
    locked = [f for f in files if not os.stat(os.path.join(root, f)).st_mode & 0o020]
    report("WARN" if locked else "PASS", "files are group-writable", f"{len(locked)} are not, e.g. {locked[0]}" if locked else "")



def check_paper_assets(root):
    need(root, "paper: reproduction scripts", ["scripts/" + name + ".py" for name in
         ("generate_paper_results", "plot_paper_results", "compute_persistence", "score_additional_metrics",
          "plot_chicago_future_precipitation", "plot_chicago_precipitation_mismatch")])
    names = ("fig01_architecture.pdf", "fig02_conditioning.png", "fig03_diagnostics.png",
             "fig04_spatial_prediction.png", "fig05_simulated_inputs.png", "fig06_observed_inputs.png",
             "fig07_four_gauges.png", "fig08_rainfall_mismatch.png", "fig09_forecast_errors.png",
             "fig10_history_length.png")
    need(root, "paper: Figures 1–10", [os.path.join("figures", name) for name in names])
    actual = {os.path.basename(p) for ext in ("*.pdf", "*.png") for p in glob.glob(os.path.join(root, "figures", ext))}
    allowed = set(names) | {"fig01_architecture.png"}
    need(root, "paper: architecture README preview", ["figures/fig01_architecture.png"])
    report("PASS" if actual == allowed else "FAIL", "paper: only published figures and architecture preview", ", ".join(sorted(actual - allowed)))
    need(root, "paper: numeric tables and forecast data", ["results/" + name for name in
         ("reconstruction_summary.csv", "baseline_metrics.csv", "gauge_simulator_metrics.csv", "usgs_metrics.csv",
          "history_common_support.csv", "table10.csv", "forecasts.json", "gauge_forecasts.json", "persistence.json",
          "precipitation_mismatch.json", "spatial_summary.json", "diagnostic_locations.json", "timing.json")])
    broken = []
    for relative in exported(root):
        if not relative.endswith(".md"):
            continue
        with open(os.path.join(root, relative), errors="replace") as stream:
            targets = re.findall(r"\]\(([^)]+)\)", stream.read())
        for target in targets:
            target = target.split("#", 1)[0]
            if target and "://" not in target and not target.startswith("mailto:"):
                if not os.path.exists(os.path.join(root, os.path.dirname(relative), target)):
                    broken.append(relative + ": " + target)
    report("FAIL" if broken else "PASS", "docs: local links", "; ".join(broken[:6]))


def check_compact_checksums(root):
    manifest = os.path.join(root, "data", "SHA256SUMS")
    if not os.path.exists(manifest):
        return
    problems = []
    listed = set()
    for line in open(manifest):
        digest, relative = line.strip().split(None, 1)
        listed.add(os.path.normpath(relative))
        path = os.path.join(root, relative)
        if not os.path.isfile(path) or sha256(path) != digest:
            problems.append(relative)
    actual = {os.path.normpath(relative) for relative in exported(root)
              if relative.startswith("data" + os.sep) and relative != os.path.join("data", "SHA256SUMS")}
    problems.extend(sorted(actual - listed))
    problems.extend(sorted(listed - actual))
    report("FAIL" if problems else "PASS", "data: compact files match SHA256SUMS",
           "; ".join(problems[:6]) if problems else str(len(listed)) + " files")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--checksums", action="store_true")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    print("checking", root)
    check_code(root)
    check_models(root, args.checksums)
    check_split(root)
    check_data(root)
    check_paper_assets(root)
    if args.checksums:
        check_compact_checksums(root)
    check_hygiene(root)
    print(f"{fails} FAIL(s)")
    sys.exit(fails)


if __name__ == "__main__":
    main()
