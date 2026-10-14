"""Check publication boundaries and consistency of released event inputs."""
import csv
import json
from pathlib import Path
import re

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_event_inputs_match_the_published_split():
    split = json.loads((ROOT / 'splits/chicago_split.json').read_text())
    ids = set(split['train'] + split['test'])
    assert len(ids) == 94 and len(split['train']) == 90 and len(split['test']) == 4
    rows = list(csv.DictReader((ROOT / 'data/storms/storm_periods.csv').open()))
    assert len(rows) == 94 and {int(row['traj_id']) for row in rows} == ids
    assert set(rows[0]) == {'traj_id', 'cldnet_event', 'start_utc', 'end_utc', 'on_basin', 'site',
                         'field_centre_lat', 'field_centre_lon', 'km_from_basin', 'mean_depth_mm',
                         'peak_mm_per_h', 'corr_with_stage4', 'corr_at_basin', 'hours_compared',
                         'hours_without_archive_data'}
    with np.load(ROOT / 'data/gauges/simulated_depth.npz', allow_pickle=False) as gauges:
        assert set(gauges['trajectory_ids'].tolist()) == ids
        assert gauges['depth_m'].shape == (94, 96, 6)


def test_no_private_identifiers_or_tracking_accounts():
    internal = {"CHECKLIST.md", "DEPOSITS.md", "private"}
    staging = ROOT.name == "cstride"
    if not staging:
        assert not any((ROOT / name).exists() for name in internal)
    patterns = [re.compile(p) for p in (r'\brun_[0-9]{8,}\b', r'\bsci[0-9]+_',
                                       r'\bSCI[- ][0-9]+', r'/(?:work[0-9]*|storage|nethome)/',
                                       r'tyj[0-9]{5,}')]
    for path in ROOT.rglob('*'):
        relative = path.relative_to(ROOT)
        if not path.is_file() or 'outputs' in relative.parts:
            continue
        if staging and relative.parts[0] in internal:
            continue
        if path.suffix in {'.py', '.md', '.json', '.yaml', '.yml', '.txt', '.csv', '.toml', '.sh'}:
            content = path.read_text(errors='replace')
            assert not any(pattern.search(content) for pattern in patterns), path.relative_to(ROOT)
        if path.suffix in {'.yaml', '.yml'}:
            config = yaml.safe_load(path.read_text())
            assert not config.get('wandb', {}).get('entity'), path.relative_to(ROOT)


def test_only_paper_figures_and_selected_results_are_bundled():
    figures = [p.name for p in (ROOT / 'figures').iterdir() if p.suffix in {'.png', '.pdf'}]
    assert len(figures) == 11
    assert {int(name[3:5]) for name in figures} == set(range(1, 11))
    assert {name for name in figures if name.startswith("fig01_")} == {"fig01_architecture.pdf", "fig01_architecture.png"}
    expected = {'README.md', 'reconstruction_summary.csv', 'history_common_support.csv',
                'gauge_simulator_metrics.csv', 'usgs_metrics.csv', 'table10.csv', 'baseline_metrics.csv',
                'forecasts.json', 'gauge_forecasts.json', 'persistence.json', 'precipitation_mismatch.json',
                'spatial_summary.json', 'diagnostic_locations.json', 'timing.json'}
    assert {p.name for p in (ROOT / 'results').iterdir() if p.is_file()} == expected


def test_plot_defaults_use_the_published_figure_directory():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import plot_paper_results
    import plot_chicago_future_precipitation
    import plot_chicago_precipitation_mismatch
    assert plot_paper_results.FIG == ROOT / "figures"
    for script in (plot_chicago_future_precipitation, plot_chicago_precipitation_mismatch):
        assert script.DEFAULT_OUTPUT.parent == ROOT / "figures"
        assert script.DEFAULT_OUTPUT.exists()
