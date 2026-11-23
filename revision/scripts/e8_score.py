"""E8 scoring: models retrained on CLDNet's fixed grid with corrected Manning and the clean split.

Normalizers are refitted on the 90 E8 training arrays exactly as the trainer does
(fit_chicago_normalizers on the E8 split). Gauge histories come from the E8 arrays at the six fixed
gauge cells. Truth for the field metrics is the dense SynxFlow depth (flow_variables[t+1], float32),
as for the released/E5 scores.
  E4  fixed grid and event masks: relative L2, RMSE, CSI/precision/recall/bias at 0.5 m, false alarms
      outside each event mask
  T7  Si et al. cross-sections, CST, frames 12-95, own-mean centring, sim and USGS-CST inputs
  E6  19 (xs) and 20 (nearest) unregulated independent gauges
Usage: e8_score.py --ckpt full=<path> --ckpt terrain=<path> ... [--tag e8]
"""
import paths as P  # noqa: E402  (input and output locations, see paths.py)
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e4_fixed_grid import Acc, CS, CLD, SIMS, EVENT_DIR, TS  # noqa: E402
from e5_score import raster_features, STATIONS, NAMES, REGULATED  # noqa: E402
from e6_independent_gauges import score, G  # noqa: E402
from e3_crosssection_scores import usgs_series, GAUGE_OF_STATION  # noqa: E402

import numpy as np
import pandas as pd
import rasterio
import torch
import yaml

from cstride.chicago import (build_sensor_sequences, load_chicago_manifest, load_chicago_sensors, make_windows,
                             chicago_include_precipitation_input, chicago_use_terrain_conditioning,
                             chicago_split_indices, fit_chicago_normalizers)
from cstride.chicago_forecast import build_chicago_feature_coords
from score_additional_metrics import encode_latent, build_feature_coords_for_entry
from cstride.models import CSTRIDE

ROOT = P.OUTPUTS
E8 = ROOT / 'e8'


def load(label, ckpt, device):
    config = yaml.safe_load((E8 / 'configs' / f'{label}_e8.yaml').read_text())
    precip = chicago_include_precipitation_input(config)
    terrain = chicago_use_terrain_conditioning(config)
    smap = load_chicago_sensors('', config)
    model = CSTRIDE(input_size=len(smap[120]) + 507 * precip, coord_size=5 if terrain else 2, output_size=1,
                    device=device, **config['model']).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    sd = {k.replace('module.', '', 1).replace('_orig_mod.', '', 1): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return model, smap, precip, terrain, int(config['basics']['lag'])


def main(args):
    device = args.device
    os.chdir(P.CSTRIDE_ROOT)  # the E8 configs give repository-relative paths, as the released ones do
    out = E8 / f'scores_{args.tag}'
    out.mkdir(parents=True, exist_ok=True)
    root, manifest, entries = load_chicago_manifest(str(E8 / 'data/Chicago'))
    idx_train, _, _ = chicago_split_indices(entries, str(E8 / 'split_e8.json'))
    norms = fit_chicago_normalizers(entries, idx_train, root)
    np.savez(out / 'normalizers_e8.npz', **{f'{k}_{s}': getattr(v, a) for k, v in norms.items()
                                           for s, a in (('min', 'min_value'), ('max', 'max_value'))})
    models = {k: load(k, v, device) for k, v in (c.split('=', 1) for c in args.ckpt)}
    grid = np.load(CLD / 'data/illinois_grid/grid.npz')
    cells = np.flatnonzero(grid['aggregate_mask'])
    shape = grid['aggregate_mask'].shape
    pos = np.full(grid['aggregate_mask'].size, -1, np.int64)
    pos[cells] = np.arange(len(cells))
    checks = dict(train_events=len(idx_train))

    def windows(entry, label, inputs=None):
        model, smap, precip, terrain, lag = models[label]
        seq = build_sensor_sequences([entry], root, smap, norms, [0], include_precipitation=precip)[0]
        if inputs is not None:
            seq = seq.clone()
            seq[:, :6] = torch.tensor(norms['depth'].transform(inputs))  # E8 gauge order = STATIONS order
        return make_windows(seq, lag)

    def decode(label, wins, feats, t):
        model, smap, precip, terrain, lag = models[label]
        with torch.inference_mode():
            latent = encode_latent(model, wins[t - lag:t - lag + 1].to(device))
            o = torch.cat([model.decode(feats[None, s:s + 200000], latent) for s in range(0, feats.shape[0], 200000)], 1)
        return norms['depth'].inverse_transform(o.cpu().numpy().reshape(-1))

    # ---- E4 ----
    rows = []
    for event in [107, 108, 109, 120]:
        entry = next(e for e in entries if e['traj_id'] == event)
        ev = np.load(CS / f'data/grids/grid_traj{event}.npz')['cell_indices']
        in_event = np.zeros(len(cells), bool)
        in_event[pos[ev[pos[ev] >= 0]]] = True
        flow = np.load(SIMS / EVENT_DIR[event] / 'flow_variables.npy', mmap_mode='r')
        acc = {(m, s): Acc() for m in models for s in ('fixed_grid', 'event_mask')}
        feats = {m: torch.as_tensor(build_feature_coords_for_entry(entry, root, norms, models[m][3]), device=device)
                 for m in models}
        wins = {m: windows(entry, m) for m in models}
        for t in TS:
            truth = np.asarray(flow[t + 1, 0], dtype=np.float32).ravel()[cells]
            for m in models:
                pr = decode(m, wins[m], feats[m], t)
                acc[m, 'fixed_grid'].add(pr, truth, ~in_event)
                acc[m, 'event_mask'].add(pr[in_event], truth[in_event])
        for (m, s), a in acc.items():
            r = dict(event=event, model=f'E8 {m}', cells=s, n_cells=int(len(cells) if s == 'fixed_grid' else in_event.sum()),
                     **a.summary())
            if s == 'fixed_grid':
                r['fp_outside_event_mask'] = a.fp_out
            rows.append(r)
        print('E4', event, flush=True)
        pd.DataFrame(rows).to_csv(out / 'e4_long.csv', index=False)

    # ---- T7 and E6 ----
    entry = next(e for e in entries if e['traj_id'] == 120)
    xs = json.loads((ROOT / 'e3/e3_crosssection_cells_nhd.json').read_text())
    gauges = [x for x in json.loads((ROOT / 'e6/e6_gauges.json').read_text()) if 'excluded' not in x]
    ser = np.load(ROOT / 'e6/e6_series.npz')
    q = [('T7', st, c[0], c[1]) for st in STATIONS for c in xs[str(GAUGE_OF_STATION[st])]['cells']]
    q += [('E6', (x['gage_index'], k), c[0], c[1]) for x in gauges for k in ('xs', 'nearest')
          for c in (x['xs_cells'] if k == 'xs' else [x['nearest_cell']])]
    qr, qc = np.array([a[2] for a in q]), np.array([a[3] for a in q])
    coords, elev, man, slo = raster_features(qr * shape[1] + qc, shape, 120, False)
    with rasterio.open(os.path.join(G, 'dem_5070.tif')) as src:
        dem = src.read(1)
    dem_q = dem[qr, qc].astype(np.float32)
    e1 = np.load(ROOT / 'e1e2/usgs_observations_CST_full.npz')
    gidx8 = np.asarray(models[next(iter(models))][1][120])
    sim8 = np.asarray(np.load(root / entry['depth_file'], mmap_mode='r')[:, gidx8, 0], dtype=np.float32)
    checks['gauge_sim_depth_equals_e1'] = float(abs(sim8 - e1['sim_depth']).max())
    rows7, rows6 = [], []
    for m in models:
        feats = torch.as_tensor(build_chicago_feature_coords(coords, elev, man, slo, norms, models[m][3]), device=device)
        for v, ins in [('sim', None), ('USGS-CST', e1['input_depth'])]:
            wins = windows(entry, m, ins)
            pred = np.asarray([decode(m, wins, feats, t) for t in TS])
            for st in STATIONS:
                sel = [i for i, a in enumerate(q) if a[0] == 'T7' and a[1] == st]
                s = pred[:, sel].max(1) + dem_q[sel].mean()
                obs = usgs_series(GAUGE_OF_STATION[st], '2013-04-17 02:00')[12:96]
                rows7.append(dict(model=f'E8 {m}', input=v, station=st, name=NAMES[STATIONS.index(st)], **score(s, obs)))
            for x in gauges:
                for k in ('xs', 'nearest'):
                    sel = [i for i, a in enumerate(q) if a[0] == 'E6' and a[1] == (x['gage_index'], k)]
                    s = pred[:, sel].max(1) + dem_q[sel].mean()
                    rows6.append(dict(model=f'E8 {m}', input=v, site_no=x['site_no'], station_nm=x['station_nm'], cells=k,
                                      valid=not (x['site_no'] in REGULATED or (x['site_no'] == '05530000' and k == 'xs')),
                                      **score(s, ser[f"USGS__{x['gage_index']}"][12:96])))
    pd.DataFrame(rows7).to_csv(out / 'table7_crosssection_long.csv', index=False)
    pd.DataFrame(rows6).to_csv(out / 'e6_long.csv', index=False)
    (out / 'checks.json').write_text(json.dumps(checks, indent=1))
    print(json.dumps(checks, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', action='append', required=True, help='label=path (full, terrain, vanilla, forcing)')
    p.add_argument('--tag', default='e8')
    p.add_argument('--device', default='cuda:0')
    main(p.parse_args())
