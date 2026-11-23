"""T1 and E2b on the Table 7 series (released sensor cells, event 120).

T1: plateau-robust timing on frames 12-95 for CST (main) and CDT (sensitivity):
    half-rise  first frame with WSE >= min + 0.5 (max - min) of that series
    centroid   centroid time of (WSE - half-rise level) over frames above that level
    reported as series minus USGS, in hours; plus r and alpha = std ratio (replacing KGE on WSE).
    Offset on 12-95 (E1); timing, r and alpha are offset-invariant.
E2b: split-sample offset (CST). Fit offset on frames 12-47, score NSE on 48-95, and the reverse.
    The offset is used both to convert USGS stage to input depth and for scoring.
"""
import paths as P  # noqa: E402  (input and output locations, see paths.py)
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e1_usgs_clock import CS, STATIONS, NAMES, CLOCKS, usgs_wse  # noqa: E402

import numpy as np
import pandas as pd
import torch
import yaml

from cstride.chicago import (MinMaxNormalizer, build_sensor_sequences, get_chicago_state_array, load_chicago_manifest,
                             load_chicago_sensors, make_windows, chicago_include_precipitation_input,
                             chicago_use_terrain_conditioning)
from score_additional_metrics import (build_feature_coords_for_entry, encode_latent, load_state_dict_compatible,
                                      nash_sutcliffe_efficiency)
from cstride.models import CSTRIDE

OUT = P.OUTPUTS
E1 = OUT / 'e1e2'
DATA = P.ARRAYS
MODELS = ['full', 'terrain', 'vanilla', 'four_gauges']


def timing(series):
    lo, hi = series.min(), series.max()
    level = lo + 0.5 * (hi - lo)
    idx = np.flatnonzero(series >= level)
    w = series[idx] - level
    centroid = float(np.sum(idx * w) / np.sum(w)) if w.sum() > 0 else float(idx.mean())
    return int(idx[0]), centroid


def t1():
    preds = np.load(E1 / 'gauge_predictions.npz')
    sensors = json.loads((E1 / 'e1e2_checks.json').read_text())
    rows = []
    for clock in ['CST', 'CDT']:
        z = np.load(E1 / f'usgs_observations_{clock}_full.npz')
        base = z['elevation'] + z['shift_m']
        obs = z['obs_wse'][12:]
        for label in MODELS:
            held = [s for s in STATIONS if s not in sensors[f'{label}_sensors']]
            srcs = [('sim_input', preds[f'{label}__sim'] + base), ('usgs_input', preds[f'{label}__{clock}__full'] + base)]
            if label == 'full':
                srcs.insert(0, ('SynxFlow', z['sim_depth'][12:] + base))
            for source, s in srcs:
                for i, st in enumerate(STATIONS):
                    hs, cs = timing(s[:, i])
                    ho, co = timing(obs[:, i])
                    rows.append(dict(clock=clock, model=label, source=source, station=st, name=NAMES[i],
                                     role='held_out' if st in held else 'input',
                                     nse=nash_sutcliffe_efficiency(s[:, i], obs[:, i]),
                                     r=float(np.corrcoef(s[:, i], obs[:, i])[0, 1]),
                                     alpha=float(s[:, i].std() / obs[:, i].std()),
                                     half_rise_time_h=hs - ho, centroid_time_h=round(cs - co, 2),
                                     obs_half_rise_frame=ho + 12, obs_centroid_frame=round(co + 12, 2)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 't1/t1_timing_long.csv', index=False)
    return df


def e2b(device='cuda:0'):
    root, manifest, entries = load_chicago_manifest(DATA)
    zn = np.load(CS / 'data/normalization/normalizers.npz')
    norms = {k: MinMaxNormalizer(zn[k + '_min'], zn[k + '_max'])
             for k in ['state', 'depth', 'precip', 'elevation', 'manning', 'slope']}
    entry = next(e for e in entries if e['traj_id'] == 120)
    gidx = np.asarray(json.load(open(CS / 'data/gauges/USGS_6sensors.json'))['120'])
    state = get_chicago_state_array(root, entry)
    sim_depth = np.asarray(state[:, gidx, 0], dtype=np.float32)
    elevation = np.load(root / entry['elevation_file'])[gidx]
    sim_wse = sim_depth + elevation
    obs = usgs_wse(CLOCKS['CST'])
    splits = {'fit12-47_score48-95': (slice(12, 48), slice(48, 96)),
              'fit48-95_score12-47': (slice(48, 96), slice(12, 48)),
              'fit12-95_score48-95': (slice(12, 96), slice(48, 96)),
              'fit12-95_score12-47': (slice(12, 96), slice(12, 48))}
    shifts = {k: obs[f].mean(0) - sim_wse[f].mean(0) for k, (f, _) in splits.items()}
    cwd = os.getcwd()
    os.chdir(CS)
    rows = []
    for label in ['full', 'four_gauges']:
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
        coords = torch.as_tensor(build_feature_coords_for_entry(entry, root, norms, terrain)[gidx], device=device).unsqueeze(0)
        positions = [int(np.flatnonzero(gidx == i)[0]) for i in smap[120]]
        held = [s for s in STATIONS if s not in [STATIONS[p] for p in positions]]

        def run(sequence):
            res = []
            with torch.inference_mode():
                for w in make_windows(sequence, lag):
                    latent = encode_latent(model, w[None].to(device))
                    res.append(norms['depth'].inverse_transform(model.decode(coords, latent).cpu().numpy().reshape(-1)))
            return np.asarray(res)

        sim_pred = run(seq)
        for k, (fit, sc) in splits.items():
            inp = np.maximum(obs - shifts[k] - elevation, 0)
            custom = seq.clone()
            custom[:, :len(positions)] = torch.tensor(norms['depth'].transform(inp[:, positions]))
            usgs_pred = run(custom)
            full96 = lambda p: np.concatenate([np.full((12, 6), np.nan), p])  # noqa: E731
            srcs = [('sim_input', full96(sim_pred)), ('usgs_input', full96(usgs_pred))]
            if label == 'full':
                srcs.insert(0, ('SynxFlow', sim_depth))
            for source, s in srcs:
                for i, st in enumerate(STATIONS):
                    series = s[sc, i] + elevation[i] + shifts[k][i]
                    rows.append(dict(split=k, model=label, source=source, station=st, name=NAMES[i],
                                     role='held_out' if st in held else 'input', offset_m=float(shifts[k][i]),
                                     nse=nash_sutcliffe_efficiency(series, obs[sc, i])))
        del model
        torch.cuda.empty_cache()
    os.chdir(cwd)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'e2b/e2b_split_offset_long.csv', index=False)
    return df


if __name__ == '__main__':
    (OUT / 't1').mkdir(exist_ok=True)
    (OUT / 'e2b').mkdir(exist_ok=True)
    t1()
    e2b()
