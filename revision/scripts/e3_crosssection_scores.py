"""E3: SynxFlow, CLDNet, LDNet and C-STRIDE vs USGS at the CLDNet notebook's cross-section cells.

Cross-section WSE = max depth over the 2-3 cells + mean dem_5070 over them (notebook definition).
Cells: rebuilt from NHD flowline_mr and checked against an exact match of the stored SynxFlow
series (results/e3/e3_crosssection_cells_nhd.json).
Clocks: frame i is UTC 2013-04-17 08:00 + i; CSV naive time is CST.
  notebook  CSV naive 07:00 + i (= CDT 08:00 + i), frames 1-95  [Si et al. Table 3 convention]
  CST       CSV naive 02:00 + i  (aligned)
  CDT       CSV naive 03:00 + i  (sensitivity)
Every series is mean-shifted to the USGS mean over the scored frames (notebook convention).
"""
import paths as P  # noqa: E402  (input and output locations, see paths.py)
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e1_usgs_clock import CS, STATIONS, NAMES  # noqa: E402  (also sets sys.path for cstride)

import numpy as np
import pandas as pd
import rasterio
import torch
import xarray as xr
import yaml

from cstride.chicago import (MinMaxNormalizer, build_sensor_sequences, load_chicago_manifest, load_chicago_sensors,
                             make_windows, chicago_include_precipitation_input, chicago_use_terrain_conditioning)
from score_additional_metrics import build_feature_coords_for_entry, encode_latent, load_state_dict_compatible
from cstride.models import CSTRIDE

G = str(P.USGS_2013)
OUT = P.OUTPUTS / 'e3'
E1 = P.OUTPUTS / 'e1e2'
DATA = P.ARRAYS
GAUGE_OF_STATION = {'05527800': 0, '05528000': 19, '05529000': 20, '05531500': 7, '05532500': 25, '05540130': 14}
W90 = [0, 19, 20, 25]
CLOCK_START = {'notebook': '2013-04-17 07:00', 'CST': '2013-04-17 02:00', 'CDT': '2013-04-17 03:00'}
H = os.path.join(G, 'USGS_2013-04-15_2013-04-23/h')
FILES = sorted(os.listdir(H), key=lambda x: int(x.split('_')[0]))


def usgs_series(g, start):
    """Notebook get_usgs_h_wse from the local CSV: hourly nearest, alt_va (+10 ft at 14, 20)."""
    d = pd.read_csv(os.path.join(H, FILES[g]))
    d['datetime'] = pd.to_datetime(d['datetime'])
    d['Gage Height (ft)'] = pd.to_numeric(d['Gage Height (ft)'], errors='coerce')
    d = d.dropna(subset=['Gage Height (ft)']).sort_values('datetime')
    times = pd.date_range(start, periods=96, freq='h')
    d = d[(d.datetime >= times[0]) & (d.datetime <= times[-1])]
    v = d[['datetime', 'Gage Height (ft)']].drop_duplicates('datetime', keep='last').set_index('datetime')
    v = v.resample('1h').nearest().reindex(times)
    alt = (float(d.alt_va.iloc[0]) + (10 if g in (14, 20) else 0)) * 0.3048
    return (v['Gage Height (ft)'] * 0.3048 + alt).to_numpy()


def timing(series):
    lo, hi = series.min(), series.max()
    level = lo + 0.5 * (hi - lo)
    above = series >= level
    half = int(np.flatnonzero(above)[0])
    w = series[above] - level
    centroid = float(np.sum(np.flatnonzero(above) * w) / np.sum(w)) if w.sum() > 0 else float(np.flatnonzero(above).mean())
    return int(np.argmax(series)), half, centroid


def score(sim, obs, sim_mean=None):
    s = sim - (sim.mean() if sim_mean is None else sim_mean) + obs.mean()
    nse = 1 - np.sum((s - obs) ** 2) / np.sum((obs - obs.mean()) ** 2)
    ps, hs, cs = timing(s)
    po, ho, co = timing(obs)
    return dict(nse=nse, r=float(np.corrcoef(s, obs)[0, 1]), alpha=float(s.std() / obs.std()),
                peak_time_h=ps - po, half_rise_time_h=hs - ho, centroid_time_h=cs - co)


def main(device='cuda:0'):
    cells = json.load(open(OUT / 'e3_crosssection_cells_nhd.json'))
    grid = np.load(CS / 'data/grids/grid_traj120.npz')
    lut = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(grid['rows'], grid['cols']))}
    with rasterio.open(os.path.join(G, 'dem_5070.tif')) as src:
        dem = src.read(1)
    truth = np.load(os.path.join(G, 'truth_traj120.npy'), mmap_mode='r')
    nc = {(m, w): xr.open_dataset(os.path.join(G, f'output/cross_section_metrics_{m}_width_{w}.nc'))
          for m in ('synxflow', 'cldnet', 'ldnet') for w in (60, 90)}

    # Cross-section series for the reference models (96 frames).
    series, cs_idx, dem_mean, checks = {}, {}, {}, {}
    for st in STATIONS:
        g = GAUGE_OF_STATION[st]
        w = 90 if g in W90 else 60
        rc = [tuple(x) for x in cells[str(g)]['cells']]
        cs_idx[st] = np.array([lut[x] for x in rc])
        rr, cc = np.array([x[0] for x in rc]), np.array([x[1] for x in rc])
        dem_mean[st] = dem[rr, cc].astype(np.float32).mean(dtype=np.float32)
        own = np.asarray(truth[:, rr, cc, 0]).max(axis=1) + dem_mean[st]
        for m, label in [('synxflow', 'SynxFlow'), ('cldnet', 'CLDNet'), ('ldnet', 'LDNet')]:
            series[label, st] = nc[m, w]['wse'].isel(gauge=g).values.astype(np.float64)
        checks[f'{st}_synxflow_recomputed_maxdiff_m'] = float(abs(own - series['SynxFlow', st]).max())
        checks[f'{st}_cells'] = rc
        checks[f'{st}_prepared_vs_dem_elevation_maxdiff_m'] = float(
            abs(np.load(DATA / 'elevation_traj120.npy')[cs_idx[st]] - dem[rr, cc]).max())

    # C-STRIDE at the cross-section cells (targets 12-95).
    root, manifest, entries = load_chicago_manifest(DATA)
    z = np.load(CS / 'data/normalization/normalizers.npz')
    norms = {k: MinMaxNormalizer(z[k + '_min'], z[k + '_max'])
             for k in ['state', 'depth', 'precip', 'elevation', 'manning', 'slope']}
    entry = next(e for e in entries if e['traj_id'] == 120)
    gidx = np.asarray(json.load(open(CS / 'data/gauges/USGS_6sensors.json'))['120'])
    all_cells = np.concatenate([cs_idx[st] for st in STATIONS])
    inputs = {c: np.load(E1 / f'usgs_observations_{c}_full.npz')['input_depth'] for c in ('CST', 'CDT')}
    cwd = os.getcwd()
    os.chdir(CS)  # config paths are relative to the release root
    for label in ['full', 'terrain', 'vanilla', 'four_gauges']:
        path = CS / 'checkpoints' / label
        config = yaml.safe_load((path / 'config.yaml').read_text())
        precip = chicago_include_precipitation_input(config)
        terrain = chicago_use_terrain_conditioning(config)
        lag = int(config['basics']['lag'])
        smap = load_chicago_sensors(path, config)
        model = CSTRIDE(input_size=len(smap[120]) + 507 * precip, coord_size=5 if terrain else 2, output_size=1,
                        device=device, **config['model']).to(device)
        model.load_state_dict(load_state_dict_compatible(path / 'best_model.ckpt', device))
        model.eval()
        seq = build_sensor_sequences([entry], root, smap, norms, [0], include_precipitation=precip)[0]
        coords = torch.as_tensor(build_feature_coords_for_entry(entry, root, norms, terrain)[all_cells],
                                 device=device).unsqueeze(0)
        positions = [int(np.flatnonzero(gidx == i)[0]) for i in smap[120]]

        def run(sequence):
            res = []
            with torch.inference_mode():
                for wdw in make_windows(sequence, lag):
                    latent = encode_latent(model, wdw[None].to(device))
                    res.append(norms['depth'].inverse_transform(model.decode(coords, latent).cpu().numpy().reshape(-1)))
            return np.asarray(res)

        variants = {'sim': seq}
        for c in ('CST', 'CDT'):
            custom = seq.clone()
            custom[:, :len(positions)] = torch.tensor(norms['depth'].transform(inputs[c][:, positions]))
            variants[c] = custom
        for vname, sq in variants.items():
            pred = run(sq)  # (84, n_cells)
            k = 0
            for st in STATIONS:
                n = len(cs_idx[st])
                full = np.full(96, np.nan)
                full[12:] = pred[:, k:k + n].max(axis=1) + dem_mean[st]
                series[f'C-STRIDE {label} ({"sim" if vname == "sim" else "USGS-" + vname} input)', st] = full
                k += n
        del model
        torch.cuda.empty_cache()
    os.chdir(cwd)

    rows = []
    for (src, st), s in series.items():
        g = GAUGE_OF_STATION[st]
        for clock, start in CLOCK_START.items():
            if '(USGS-' in src and src.split('USGS-')[1][:3] != clock:
                continue  # score USGS-input runs only on their own clock
            obs = usgs_series(g, start)
            for window, frames in [('all', np.arange(1 if clock == 'notebook' else 0, 96)), ('12-95', np.arange(12, 96))]:
                if np.isnan(s[frames]).any():
                    continue
                # The notebook centred each simulated series on all 96 frames but scored frames 1-95.
                sim_mean = s.mean() if (clock == 'notebook' and window == 'all') else None
                rows.append(dict(source=src, station=st, name=NAMES[STATIONS.index(st)], clock=clock, frames=window,
                                 n=len(frames), **score(s[frames], obs[frames], sim_mean)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'e3_crosssection_scores_long.csv', index=False)
    (OUT / 'e3_checks.json').write_text(json.dumps(checks, indent=1, default=str))
    np.savez_compressed(OUT / 'e3_crosssection_series.npz', **{f'{a}__{b}': v for (a, b), v in series.items()})
    print(json.dumps(checks, indent=1, default=str))


if __name__ == '__main__':
    main()
