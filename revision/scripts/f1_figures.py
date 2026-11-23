"""F1: redraw Figs 5-7 (simulated inputs, observed inputs, four gauges) on the CST clock with the
released plotting code (scripts/plot_paper_results.py: gauge_panels), redirected to new folders.

A  released sensor cells, Table 7 convention (SynxFlow offset on 12-95 applied to all series)
B  notebook cross-section cells, each series centred on its own 12-95 mean (fed to the
   released code as pre-centred WSE with zero elevation and offset)
Time axis: simulation clock (UTC 2013-04-17 08:00 + i); USGS paired on CST (CSV 02:00 + i).
Map markers in Fig 5 are the released sensor cells in both variants. Nothing in the release is
written: the module's OUT and FIG are pointed at results/figures/{A,B}.
"""
import paths as P  # noqa: E402  (input and output locations, see paths.py)
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CS = P.CSTRIDE_ROOT
sys.path.insert(0, str(CS / 'code'))
sys.path.insert(0, str(CS / 'scripts'))
import numpy as np  # noqa: E402
import plot_paper_results as ppr  # noqa: E402

PAPER = P.PAPER_RESULTS
E1 = P.OUTPUTS / 'e1e2'
E3 = P.OUTPUTS / 'e3'
FIGS = P.OUTPUTS / 'figures'
STATIONS = ppr.STATIONS
times = np.asarray([str(t) for t in np.arange(np.datetime64('2013-04-17T08:00'), np.datetime64('2013-04-21T08:00'),
                                              np.timedelta64(1, 'h'))])


def stage(variant):
    d = FIGS / variant / 'inputs'
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(PAPER / 'reference_traj120.npz', d / 'reference_traj120.npz')
    obs = np.load(E1 / 'usgs_observations_CST_full.npz')
    preds = np.load(E1 / 'gauge_predictions.npz')
    if variant == 'A':
        np.savez(d / 'usgs_observations.npz', obs_wse=obs['obs_wse'], shift_m=obs['shift_m'], elevation=obs['elevation'],
                 sim_depth=obs['sim_depth'], input_depth=obs['input_depth'], times=times)
        for key in ['full', 'four_gauges']:
            old = np.load(PAPER / f'{key}_traj120.npz')
            np.savez(d / f'{key}_traj120.npz', point_predictions=old['point_predictions'],
                     usgs_input_predictions=preds[f'{key}__CST__full'], snapshot_prediction=old['snapshot_prediction'])
    else:
        xs = np.load(E3 / 'e3_crosssection_series.npz')
        o = obs['obs_wse']
        centred = lambda s, i: s - np.nanmean(s[12:]) + o[12:, i].mean()  # noqa: E731
        sim = np.column_stack([centred(xs[f'SynxFlow__{st}'], i) for i, st in enumerate(STATIONS)])
        np.savez(d / 'usgs_observations.npz', obs_wse=o, shift_m=np.zeros(6), elevation=np.zeros(6), sim_depth=sim,
                 times=times)
        for key in ['full', 'four_gauges']:
            old = np.load(PAPER / f'{key}_traj120.npz')
            p = np.column_stack([centred(xs[f'C-STRIDE {key} (sim input)__{st}'], i)[12:] for i, st in enumerate(STATIONS)])
            u = np.column_stack([centred(xs[f'C-STRIDE {key} (USGS-CST input)__{st}'], i)[12:] for i, st in enumerate(STATIONS)])
            np.savez(d / f'{key}_traj120.npz', point_predictions=p, usgs_input_predictions=u,
                     snapshot_prediction=old['snapshot_prediction'])
    return d


for variant in ['A', 'B']:
    ppr.OUT = stage(variant)
    ppr.FIG = FIGS / variant
    ppr.PREFIX = ''
    ref = np.load(ppr.OUT / 'reference_traj120.npz')
    ppr.gauge_panels(ref, 'full', True)
    ppr.gauge_panels(ref, 'full')
    ppr.gauge_panels(ref, 'four_gauges')
