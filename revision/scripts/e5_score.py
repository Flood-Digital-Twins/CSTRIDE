"""E5 scoring: C-STRIDE full and terrain retrained with the corrected Manning channel, scored with
corrected Manning features (prepare_arrays.py rule without --legacy-manning-order).

  E4   fixed CLDNet grid (1,408,587 cells) and each event's >= 0.1 m mask; relative L2, RMSE,
       CSI/precision/recall/bias at 0.5 m (pooled over frames 12-95)
  T7   Si et al. cross-sections (results/e3 cells), CST, frames 12-95, own-mean centring,
       sim and USGS-CST inputs; NSE, r, alpha, half-rise and centroid timing
  E6   the 19 (xs) and 20 (nearest) unregulated independent gauges, same protocol as E6

Usage: e5_score.py --full <best_model.ckpt> --terrain <best_model.ckpt> [--tag retrained]
Passing the released checkpoints with --legacy reproduces E4/E3/E6 (sanity check).
"""
import paths as P  # noqa: E402  (input and output locations, see paths.py)
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e4_fixed_grid import Acc, prepare_arrays, CS, CLD, SIMS, EVENT_DIR, TS  # noqa: E402
from e6_independent_gauges import score, G  # noqa: E402
from e3_crosssection_scores import usgs_series, GAUGE_OF_STATION  # noqa: E402

import numpy as np
import pandas as pd
import rasterio
import torch
import yaml

from cstride.chicago import (MinMaxNormalizer, build_sensor_sequences, load_chicago_manifest, load_chicago_sensors,
                             make_windows, chicago_include_precipitation_input, chicago_use_terrain_conditioning)
from cstride.chicago_forecast import build_chicago_feature_coords
from score_additional_metrics import encode_latent
from cstride.models import CSTRIDE

ROOT = P.OUTPUTS
E5 = ROOT / 'e5'
DATA = {False: E5 / 'data/Chicago', True: P.ARRAYS}
STATIONS = ['05527800', '05528000', '05529000', '05531500', '05532500', '05540130']
NAMES = ['Russell', 'Gurnee', 'Des Plaines', 'Salt Creek', 'Riverside', 'DuPage']
REGULATED = {'05536995', '05537980'}
_RASTER = {}


def raster_features(cells, shape, event, legacy):
    """prepare_arrays.py features for arbitrary raster cells (historical precision)."""
    key = legacy
    if key not in _RASTER:
        _RASTER[key] = prepare_arrays.raw_static(SIMS / EVENT_DIR[107], legacy)
    elev, valid, mean, std, gx, gy, manning = _RASTER[key]
    rows, cols = np.unravel_index(cells, shape)
    height, width = shape
    x = 2 * (np.arange(width, dtype=np.float32) / width - .5)
    y = 2 * (np.arange(height, dtype=np.float32) / height - .5) * (height / float(width))
    coords = np.column_stack([x[cols], y[rows]]).astype(np.float16)
    physical = ((elev - mean) / std).astype(np.float16).astype(np.float32) * std + mean
    rough = (manning * 100).astype(np.float16).astype(np.float32) / 100
    slope = np.sqrt(gx ** 2 + gy ** 2).astype(np.float16).astype(np.float32)
    if event == 120:
        physical = elev
        slope = np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)
    return coords, physical.ravel()[cells], rough.ravel()[cells], slope.ravel()[cells]


def load(label, ckpt, device):
    config = yaml.safe_load((CS / 'checkpoints' / label / 'config.yaml').read_text())
    precip = chicago_include_precipitation_input(config)
    terrain = chicago_use_terrain_conditioning(config)
    smap = load_chicago_sensors(CS / 'checkpoints' / label, config)
    model = CSTRIDE(input_size=len(smap[120]) + 507 * precip, coord_size=5 if terrain else 2, output_size=1,
                    device=device, **config['model']).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    sd = {k.replace('module.', '', 1).replace('_orig_mod.', '', 1): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return model, smap, precip, terrain, int(config['basics']['lag'])


def main(args):
    device = args.device
    out = ROOT / f'e5/scores_{args.tag}'
    out.mkdir(parents=True, exist_ok=True)
    legacy = args.legacy
    root, manifest, entries = load_chicago_manifest(DATA[legacy])
    z = np.load(CS / 'data/normalization/normalizers.npz')
    norms = {k: MinMaxNormalizer(z[k + '_min'], z[k + '_max'])
             for k in ['state', 'depth', 'precip', 'elevation', 'manning', 'slope']}
    os.chdir(CS)
    models = {'full': load('full', args.full, device), 'terrain': load('terrain', args.terrain, device)}
    models.update({k: load(k, v, device) for k, v in (c.split('=', 1) for c in args.extra)})  # e.g. vanilla=<ckpt>
    grid = np.load(CLD / 'data/illinois_grid/grid.npz')
    mask = grid['aggregate_mask']
    shape = mask.shape
    cells = np.flatnonzero(mask)
    pos = np.full(mask.size, -1, np.int64)
    pos[cells] = np.arange(len(cells))
    checks = {}

    def windows(entry, label, inputs=None):
        model, smap, precip, terrain, lag = models[label]
        seq = build_sensor_sequences([entry], root, smap, norms, [0], include_precipitation=precip)[0]
        if inputs is not None:
            gidx = np.asarray(json.load(open(CS / 'data/gauges/USGS_6sensors.json'))['120'])
            positions = [int(np.flatnonzero(gidx == i)[0]) for i in smap[120]]
            seq = seq.clone()
            seq[:, :len(positions)] = torch.tensor(norms['depth'].transform(inputs[:, positions]))
        return make_windows(seq, lag)

    def decode(label, wins, feats, t):
        model, smap, precip, terrain, lag = models[label]
        with torch.inference_mode():
            latent = encode_latent(model, wins[t - lag:t - lag + 1].to(device))
            o = torch.cat([model.decode(feats[None, s:s + 200000], latent) for s in range(0, feats.shape[0], 200000)], 1)
        return norms['depth'].inverse_transform(o.cpu().numpy().reshape(-1))

    # ---- E4 -------------------------------------------------------------------------------
    rows = []
    for event in [107, 108, 109, 120]:
        entry = next(e for e in entries if e['traj_id'] == event)
        coords, elev, man, slo = raster_features(cells, shape, event, legacy)
        ev = np.load(CS / f'data/grids/grid_traj{event}.npz')['cell_indices']
        inside = pos[ev] >= 0
        p = pos[ev[inside]]
        checks[f'E4_{event}_manning_vs_prepared_maxdiff'] = float(abs(man[p] - np.load(root / entry['manning_file'])[inside]).max())
        checks[f'E4_{event}_elevation_vs_prepared_maxdiff'] = float(abs(elev[p] - np.load(root / entry['elevation_file'])[inside]).max())
        in_event = np.zeros(len(cells), bool)
        in_event[p] = True
        flow = np.load(SIMS / EVENT_DIR[event] / 'flow_variables.npy', mmap_mode='r')
        acc = {(m, s): Acc() for m in models for s in ('fixed_grid', 'event_mask')}
        feats = {m: torch.as_tensor(build_chicago_feature_coords(coords, elev, man, slo, norms, models[m][3]), device=device)
                 for m in models}
        wins = {m: windows(entry, m) for m in models}
        for t in TS:
            truth = np.asarray(flow[t + 1, 0], dtype=np.float32).ravel()[cells]
            for m in models:
                pr = decode(m, wins[m], feats[m], t)
                acc[m, 'fixed_grid'].add(pr, truth, ~in_event)
                acc[m, 'event_mask'].add(pr[in_event], truth[in_event])
        for (m, s), a in acc.items():
            r = dict(event=event, model=f'C-STRIDE {m}', cells=s, n_cells=int(len(cells) if s == 'fixed_grid' else in_event.sum()),
                     **a.summary())
            if s == 'fixed_grid':
                r['fp_outside_event_mask'] = a.fp_out
            rows.append(r)
        print('E4', event, flush=True)
    pd.DataFrame(rows).to_csv(out / 'e4_long.csv', index=False)

    # ---- Table 7 (cross-sections) and E6 ---------------------------------------------------
    entry = next(e for e in entries if e['traj_id'] == 120)
    xs = json.loads((ROOT / 'e3/e3_crosssection_cells_nhd.json').read_text())
    gauges = [x for x in json.loads((ROOT / 'e6/e6_gauges.json').read_text()) if 'excluded' not in x]
    ser = np.load(ROOT / 'e6/e6_series.npz')
    q = [('T7', st, c[0], c[1]) for st in STATIONS for c in xs[str(GAUGE_OF_STATION[st])]['cells']]
    q += [('E6', (x['gage_index'], k), c[0], c[1]) for x in gauges for k in ('xs', 'nearest')
          for c in (x['xs_cells'] if k == 'xs' else [x['nearest_cell']])]
    qr, qc = np.array([a[2] for a in q]), np.array([a[3] for a in q])
    coords, elev, man, slo = raster_features(qr * shape[1] + qc, shape, 120, legacy)
    with rasterio.open(os.path.join(G, 'dem_5070.tif')) as src:
        dem = src.read(1)
    dem_q = dem[qr, qc].astype(np.float32)
    inp = np.load(ROOT / 'e1e2/usgs_observations_CST_full.npz')['input_depth']
    rows7, rows6 = [], []
    for m in models:
        feats = torch.as_tensor(build_chicago_feature_coords(coords, elev, man, slo, norms, models[m][3]), device=device)
        for v, ins in [('sim', None), ('USGS-CST', inp)]:
            wins = windows(entry, m, ins)
            pred = np.asarray([decode(m, wins, feats, t) for t in TS])  # (84, nq)
            for st in STATIONS:
                sel = [i for i, a in enumerate(q) if a[0] == 'T7' and a[1] == st]
                s = pred[:, sel].max(1) + dem_q[sel].mean()
                obs = usgs_series(GAUGE_OF_STATION[st], '2013-04-17 02:00')[12:96]
                rows7.append(dict(model=m, input=v, station=st, name=NAMES[STATIONS.index(st)], **score(s, obs)))
            for x in gauges:
                for k in ('xs', 'nearest'):
                    sel = [i for i, a in enumerate(q) if a[0] == 'E6' and a[1] == (x['gage_index'], k)]
                    s = pred[:, sel].max(1) + dem_q[sel].mean()
                    rows6.append(dict(model=m, input=v, site_no=x['site_no'], station_nm=x['station_nm'], cells=k,
                                      valid=not (x['site_no'] in REGULATED or (x['site_no'] == '05530000' and k == 'xs')),
                                      **score(s, ser[f"USGS__{x['gage_index']}"][12:96])))
    pd.DataFrame(rows7).to_csv(out / 'table7_crosssection_long.csv', index=False)
    pd.DataFrame(rows6).to_csv(out / 'e6_long.csv', index=False)
    (out / 'checks.json').write_text(json.dumps(checks, indent=1))
    print(json.dumps(checks, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--full', required=True)
    p.add_argument('--terrain', required=True)
    p.add_argument('--tag', default='retrained')
    p.add_argument('--legacy', action='store_true', help='legacy Manning features (released checkpoints)')
    p.add_argument('--extra', action='append', default=[], help='label=ckpt for other released models (vanilla, forcing)')
    p.add_argument('--device', default='cuda:0')
    main(p.parse_args())
