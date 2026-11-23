"""E4 (R2C3): score C-STRIDE full, C-STRIDE terrain and CLDNet epoch 539 on CLDNet's fixed
1,408,587-cell Des Plaines grid (grid.npz aggregate_mask) for events 107, 108, 109 and 120.

C-STRIDE features on the fixed grid follow prepare_arrays.py exactly: legacy Manning order,
historical precision (quantized elevation/slope for numbered events, raw float32 for 120), and
the C-STRIDE coordinate formula (identical to CLDNet's stored coords). They are checked cell by
cell against the released grid_traj<e>.npz where the grids overlap.
Truth: dense SynxFlow flow_variables[t + 1] (burn-in 1), depth channel, frames t = 12..95.
Metrics (Tables 5 and 8 conventions): mean snapshot-wise relative L2 and RMSE; CSI, precision,
recall, bias from counts pooled over frames at the 0.5 m inundation threshold. False alarms
outside each event's >= 0.1 m mask are counted separately.
"""
import paths as P  # noqa: E402  (input and output locations, see paths.py)
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e1_usgs_clock import CS  # noqa: E402  (sets sys.path for cstride)

import numpy as np
import pandas as pd
import torch
import yaml

from cstride.chicago import (MinMaxNormalizer, build_sensor_sequences, load_chicago_manifest, load_chicago_sensors,
                             make_windows, chicago_include_precipitation_input, chicago_use_terrain_conditioning)
from cstride.chicago_forecast import build_chicago_feature_coords
from score_additional_metrics import encode_latent, load_state_dict_compatible
from cstride.models import CSTRIDE

spec = importlib.util.spec_from_file_location('prepare_arrays', CS / 'preprocessing/prepare_arrays/prepare_arrays.py')
prepare_arrays = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_arrays)

OUT = P.OUTPUTS / 'e4'
DATA = P.ARRAYS
CLD = P.CLDNET
SIMS = P.SIMS
EVENT_DIR = {107: 'event_107_filling_hours_0_filling_intensity_mm_hr_0',
             108: 'event_108_filling_hours_0_filling_intensity_mm_hr_0',
             109: 'event_109_filling_hours_0_filling_intensity_mm_hr_0',
             120: 'event_2013-04-10_2013-04-23_manning_water_0.02_manning_land_0.05'}
CLDNET_FILE = {107: 'cldnet_107.npz', 108: 'cldnet_108.npz', 109: 'cldnet_109.npz', 120: 'cldnet_2013-04-17_2013-04-21.npz'}
THRESH = 0.5
TS = np.arange(12, 96)


def fixed_grid_features(event, cells, shape):
    """prepare_arrays.py feature construction, evaluated on arbitrary raster cells."""
    elev, valid, mean, std, gx, gy, manning = prepare_arrays.raw_static(SIMS / EVENT_DIR[event], True)
    rows, cols = np.unravel_index(cells, shape)
    height, width = shape
    x = 2 * (np.arange(width, dtype=np.float32) / width - .5)
    y = 2 * (np.arange(height, dtype=np.float32) / height - .5) * (height / float(width))
    coords = np.column_stack([x[cols], y[rows]]).astype(np.float16)
    z = ((elev - mean) / std).astype(np.float16)
    physical = z.astype(np.float32) * std + mean
    roughness = (manning * 100).astype(np.float16).astype(np.float32) / 100
    slope = np.sqrt(gx ** 2 + gy ** 2).astype(np.float16).astype(np.float32)
    if event == 120:
        physical = elev
        slope = np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)
    return coords, physical.ravel()[cells], roughness.ravel()[cells], slope.ravel()[cells]


class Acc:
    def __init__(self):
        self.rel, self.rmse, self.counts, self.fp_out = [], [], np.zeros(3, np.int64), 0

    def add(self, pred, truth, outside=None):
        sq = np.square(pred.astype(np.float64) - truth)
        self.rel.append(float(np.sqrt(sq.sum() / np.square(truth.astype(np.float64)).sum())))
        self.rmse.append(float(np.sqrt(sq.mean())))
        a, b = pred >= THRESH, truth >= THRESH
        self.counts += [np.count_nonzero(a & b), np.count_nonzero(a & ~b), np.count_nonzero(~a & b)]
        if outside is not None:
            self.fp_out += int(np.count_nonzero(a[outside] & ~b[outside]))

    def summary(self):
        tp, fp, fn = self.counts
        return dict(mean_relative_l2=float(np.mean(self.rel)), mean_rmse_m=float(np.mean(self.rmse)),
                    csi=tp / (tp + fp + fn), precision=tp / (tp + fp), recall=tp / (tp + fn), bias=(tp + fp) / (tp + fn),
                    tp=int(tp), fp=int(fp), fn=int(fn))


def main(device='cuda:0'):
    OUT.mkdir(parents=True, exist_ok=True)
    grid = np.load(CLD / 'data/illinois_grid/grid.npz')
    mask = grid['aggregate_mask']
    shape = mask.shape
    cells = np.flatnonzero(mask)
    pos = np.full(mask.size, -1, np.int64)
    pos[cells] = np.arange(len(cells))
    root, manifest, entries = load_chicago_manifest(DATA)
    z = np.load(CS / 'data/normalization/normalizers.npz')
    norms = {k: MinMaxNormalizer(z[k + '_min'], z[k + '_max'])
             for k in ['state', 'depth', 'precip', 'elevation', 'manning', 'slope']}
    models = {}
    os.chdir(CS)
    for label in ['full', 'terrain']:
        path = CS / 'checkpoints' / label
        config = yaml.safe_load((path / 'config.yaml').read_text())
        precip = chicago_include_precipitation_input(config)
        smap = load_chicago_sensors(path, config)
        model = CSTRIDE(input_size=len(smap[120]) + 507 * precip, coord_size=5, output_size=1, device=device,
                        **config['model']).to(device)
        model.load_state_dict(load_state_dict_compatible(path / 'best_model.ckpt', device))
        model.eval()
        assert chicago_use_terrain_conditioning(config)
        models[label] = (model, smap, precip, int(config['basics']['lag']))

    checks, rows, perframe = {}, [], {}
    for event in [107, 108, 109, 120]:
        entry = next(e for e in entries if e['traj_id'] == event)
        coords, elev, man, slo = fixed_grid_features(event, cells, shape)
        g = np.load(CS / f'data/grids/grid_traj{event}.npz')
        ev_cells = g['cell_indices']
        inside = pos[ev_cells] >= 0
        p = pos[ev_cells[inside]]
        checks[event] = dict(
            event_cells=int(len(ev_cells)), event_cells_outside_fixed_grid=int((~inside).sum()),
            coords_equal=bool(np.array_equal(coords[p], g['coords'][inside])),
            elevation_maxdiff=float(abs(elev[p] - g['elevation'][inside]).max()),
            manning_maxdiff=float(abs(man[p] - g['manning'][inside]).max()),
            slope_maxdiff=float(abs(slo[p] - g['slope'][inside]).max()),
            prepared_elevation_equal=bool(np.array_equal(np.load(root / entry['elevation_file'])[inside], g['elevation'][inside])))
        in_event = np.zeros(len(cells), bool)
        in_event[p] = True
        outside = ~in_event
        flow = np.load(SIMS / EVENT_DIR[event] / 'flow_variables.npy', mmap_mode='r')
        cld = np.load(OUT / 'cldnet' / CLDNET_FILE[event])
        assert np.array_equal(cld['cell_index'], cells)
        cld_h = cld['fields'][..., 0]
        acc = {(m, s): Acc() for m in ['C-STRIDE full', 'C-STRIDE terrain', 'CLDNet'] for s in ['fixed_grid', 'event_mask']}
        feats, wins = {}, {}
        for label, (model, smap, precip, lag) in models.items():
            feats[label] = torch.as_tensor(build_chicago_feature_coords(coords, elev, man, slo, norms, True), device=device)
            seq = build_sensor_sequences([entry], root, smap, norms, [0], include_precipitation=precip)[0]
            wins[label] = make_windows(seq, lag)
        for t in TS:
            truth = np.asarray(flow[t + 1, 0], dtype=np.float32).ravel()[cells]
            preds = {'CLDNet': cld_h[t].astype(np.float32)}
            with torch.inference_mode():
                for label, (model, smap, precip, lag) in models.items():
                    latent = encode_latent(model, wins[label][t - lag:t - lag + 1].to(device))
                    out = torch.cat([model.decode(feats[label][None, s:s + 200000], latent)
                                     for s in range(0, len(cells), 200000)], dim=1)
                    preds[f'C-STRIDE {label}'] = norms['depth'].inverse_transform(out.cpu().numpy().reshape(-1))
            for m, pr in preds.items():
                acc[m, 'fixed_grid'].add(pr, truth, outside)
                acc[m, 'event_mask'].add(pr[in_event], truth[in_event])
        for (m, s), a in acc.items():
            r = dict(event=event, model=m, cells=s, n_cells=int(len(cells) if s == 'fixed_grid' else in_event.sum()), **a.summary())
            if s == 'fixed_grid':
                r['fp_outside_event_mask'] = a.fp_out
                r['fp_outside_share_of_fp'] = a.fp_out / max(r['fp'], 1)
            rows.append(r)
            perframe[f'{m}__{s}__{event}'] = np.asarray(a.rel)
        print(event, json.dumps(checks[event]), flush=True)
        pd.DataFrame(rows).to_csv(OUT / 'e4_fixed_grid_long.csv', index=False)
    (OUT / 'e4_checks.json').write_text(json.dumps(checks, indent=1))
    np.savez_compressed(OUT / 'e4_perframe_relative_l2.npz', **perframe)


if __name__ == '__main__':
    main()
