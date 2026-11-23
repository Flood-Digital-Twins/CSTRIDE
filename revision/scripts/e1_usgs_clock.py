"""E1/E2: USGS comparison for the April 2013 event with corrected clocks.

Simulation frame i is UTC 2013-04-17 08:00 + i. The raw_stage CSVs are naive CST
(UTC-6; checked sample-by-sample against api.waterdata.usgs.gov for all six gauges).
Clock options give the naive CSV time paired with frame 0:
  published  2013-04-17 08:00 (the paper's pairing; sim about 6 h late)
  CST        2013-04-17 02:00 (correct)
  CDT        2013-04-17 03:00 (sensitivity)
Offsets (USGS-to-simulator datum) are recomputed for each clock:
  full    mean over frames 12-95 (paper convention, retrospective)
  causal  mean over frames 0-11 (only the history before the first target)
Rainfall inputs are left on the UTC simulation clock.

Reads the released code and checkpoints (read-only); writes only under OUT.
"""
import paths as P  # noqa: E402  (input and output locations, see paths.py)
import argparse
import json
import sys
from pathlib import Path

CS = P.CSTRIDE_ROOT
sys.path.insert(0, str(CS / 'code'))
sys.path.insert(0, str(CS / 'scripts'))

import numpy as np
import pandas as pd
import torch
import yaml

from cstride.chicago import (MinMaxNormalizer, build_sensor_sequences, get_chicago_state_array,
                             load_chicago_manifest, load_chicago_sensors, load_sensor_map_file, make_windows,
                             chicago_include_precipitation_input, chicago_use_terrain_conditioning)
from score_additional_metrics import (build_feature_coords_for_entry, encode_latent, load_state_dict_compatible,
                                      nash_sutcliffe_efficiency, kling_gupta_efficiency)
from cstride.models import CSTRIDE

PAPER = P.PAPER_RESULTS
STATIONS = ['05527800', '05528000', '05529000', '05531500', '05532500', '05540130']
NAMES = ['Russell', 'Gurnee', 'Des Plaines', 'Salt Creek', 'Riverside', 'DuPage']
CLOCKS = {'published': '2013-04-17 08:00', 'CST': '2013-04-17 02:00', 'CDT': '2013-04-17 03:00'}
MODELS = ['full', 'terrain', 'forcing', 'vanilla', 'four_gauges']


def usgs_wse(start):
    """Hourly USGS WSE for 96 frames starting at naive CSV time `start` (observations() conventions)."""
    times = pd.date_range(start, periods=96, freq='h')
    cols = []
    for station in STATIONS:
        matches = []
        for path in (CS / 'data/gauges/usgs_2013/raw_stage').glob('*_height.csv'):
            d = pd.read_csv(path, dtype={'site_no': str})
            if len(d) and d.site_no.iloc[0].zfill(8) == station:
                matches.append((path, d))
        assert len(matches) == 1
        path, d = matches[0]
        datum = float(d.alt_va.iloc[0])
        manual = 10. if int(path.name.split('_')[0]) in (14, 20) else 0.
        d['datetime'] = pd.to_datetime(d.datetime)
        d['wse'] = .3048 * (pd.to_numeric(d['Gage Height (ft)'], errors='coerce') + datum + manual)
        d = d.dropna(subset=['wse']).drop_duplicates('datetime', keep='last').set_index('datetime').sort_index()
        d = d.loc[times[0]:times[-1], 'wse'].resample('h').nearest().reindex(times)
        assert d.notna().all(), (station, start)
        cols.append(d.to_numpy())
    return np.column_stack(cols)


def shift_for(obs, sim_wse, mode):
    sl = slice(12, 96) if mode == 'full' else slice(0, 12)
    return obs[sl].mean(0) - sim_wse[sl].mean(0)


def metrics(series, reference):
    return dict(nse=nash_sutcliffe_efficiency(series, reference), kge=kling_gupta_efficiency(series, reference),
                peak_timing_hours=int(np.argmax(series) - np.argmax(reference)))


def main(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    root, manifest, entries = load_chicago_manifest(args.data_root)
    z = np.load(CS / 'data/normalization/normalizers.npz')
    norms = {k: MinMaxNormalizer(z[k + '_min'], z[k + '_max'])
             for k in ['state', 'depth', 'precip', 'elevation', 'manning', 'slope']}
    entry = next(e for e in entries if e['traj_id'] == 120)
    gidx = np.asarray(load_sensor_map_file(CS / 'data/gauges/USGS_6sensors.json')[120])
    state = get_chicago_state_array(root, entry)
    sim_depth = np.asarray(state[:, gidx, 0], dtype=np.float32)
    elevation = np.load(root / entry['elevation_file'])[gidx]
    sim_wse = sim_depth + elevation

    # Reproduction check against the published products.
    pub = np.load(PAPER / 'usgs_observations.npz')
    obs_pub = usgs_wse(CLOCKS['published'])
    checks = dict(sim_depth_maxdiff=float(abs(sim_depth - pub['sim_depth']).max()),
                  elevation_maxdiff=float(abs(elevation - pub['elevation']).max()),
                  obs_wse_maxdiff=float(abs(obs_pub - pub['obs_wse']).max()),
                  shift_maxdiff=float(abs(shift_for(obs_pub, sim_wse, 'full') - pub['shift_m']).max()))

    # Observation sets.
    obs = {c: usgs_wse(s) for c, s in CLOCKS.items()}
    variants = [(c, m) for c in CLOCKS for m in ('full', 'causal')]
    shifts = {(c, m): shift_for(obs[c], sim_wse, m) for c, m in variants}
    inputs = {(c, m): np.maximum(obs[c] - shifts[c, m] - elevation, 0) for c, m in variants}
    checks['input_depth_maxdiff'] = float(abs(inputs['published', 'full'] - pub['input_depth']).max())
    for (c, m), v in inputs.items():
        np.savez_compressed(out / f'usgs_observations_{c}_{m}.npz', obs_wse=obs[c], shift_m=shifts[c, m],
                            input_depth=v, elevation=elevation, sim_depth=sim_depth,
                            times_csv_naive=pd.date_range(CLOCKS[c], periods=96, freq='h').astype(str).to_numpy(dtype=str),
                            times_sim_utc=pd.date_range('2013-04-17 08:00', periods=96, freq='h').astype(str).to_numpy(dtype=str))

    # Predictions at the six gauge cells.
    preds = {}
    for label in MODELS:
        path = CS / 'checkpoints' / label
        config = yaml.safe_load((path / 'config.yaml').read_text())
        precip = chicago_include_precipitation_input(config)
        terrain = chicago_use_terrain_conditioning(config)
        lag = int(config['basics']['lag'])
        smap = load_chicago_sensors(path, config)
        model = CSTRIDE(input_size=len(smap[120]) + 507 * precip, coord_size=5 if terrain else 2, output_size=1,
                        device=args.device, **config['model']).to(args.device)
        model.load_state_dict(load_state_dict_compatible(path / 'best_model.ckpt', args.device))
        model.eval()
        seq = build_sensor_sequences([entry], root, smap, norms, [0], include_precipitation=precip)[0]
        coords = torch.as_tensor(build_feature_coords_for_entry(entry, root, norms, terrain)[gidx],
                                 device=args.device).unsqueeze(0)
        positions = [int(np.flatnonzero(gidx == i)[0]) for i in smap[120]]

        def run(sequence):
            res = []
            with torch.inference_mode():
                for w in make_windows(sequence, lag):
                    latent = encode_latent(model, w[None].to(args.device))
                    res.append(norms['depth'].inverse_transform(model.decode(coords, latent).cpu().numpy().reshape(-1)))
            return np.asarray(res)  # (96-lag, 6), targets lag..95

        preds[label, 'sim'] = run(seq)
        for c, m in variants:
            custom = seq.clone()
            custom[:, :len(positions)] = torch.tensor(norms['depth'].transform(inputs[c, m][:, positions]))
            preds[label, c, m] = run(custom)
        if (PAPER / f'{label}_traj120.npz').exists():
            ref = np.load(PAPER / f'{label}_traj120.npz')
            checks[f'{label}_sim_input_maxdiff_m'] = float(abs(preds[label, 'sim'] - ref['point_predictions'][:, :6]).max())
            if ref['usgs_input_predictions'].size:
                checks[f'{label}_usgs_input_maxdiff_m'] = float(
                    abs(preds[label, 'published', 'full'] - ref['usgs_input_predictions']).max())
        checks[f'{label}_sensors'] = [STATIONS[p] for p in positions]
        del model
        torch.cuda.empty_cache()
    np.savez_compressed(out / 'gauge_predictions.npz', **{'__'.join(k): v for k, v in preds.items()})

    # Metrics on frames 12-95, evaluation offset = the variant's offset (E1 full, E2 causal);
    # E2 is also scored with the full-window offset to separate input and scoring effects.
    rows = []
    for c, m in variants:
        for score_mode in ([m] if m == 'full' else ['causal', 'full']):
            sh = shifts[c, score_mode]
            for label in MODELS:
                held = [] if label != 'four_gauges' else [s for s in STATIONS if s not in checks[f'{label}_sensors']]
                for i in range(6):
                    reference = obs[c][12:, i]
                    base = elevation[i] + sh[i]
                    srcs = [('sim_input', preds[label, 'sim'][:, i] + base),
                            ('usgs_input', preds[label, c, m][:, i] + base)]
                    if label == 'full':
                        srcs.insert(0, ('SynxFlow', sim_depth[12:, i] + base))
                    for source, series in srcs:
                        rows.append(dict(clock=c, input_offset=m, scoring_offset=score_mode, model=label,
                                         source=source, station=STATIONS[i], name=NAMES[i],
                                         role='held_out' if STATIONS[i] in held else 'input',
                                         offset_m=float(sh[i]), **metrics(series, reference)))
    df = pd.DataFrame(rows)
    df.to_csv(out / 'e1e2_usgs_metrics_long.csv', index=False)

    # SynxFlow-only clock scan (lag in hours of the CSV relative to the published pairing).
    scan = []
    for hours in range(-10, 3):
        start = pd.Timestamp(CLOCKS['published']) + pd.Timedelta(hours=hours)
        o = usgs_wse(start)
        sh = shift_for(o, sim_wse, 'full')
        nse = [nash_sutcliffe_efficiency(sim_wse[12:, i] + sh[i], o[12:, i]) for i in range(6)]
        scan.append(dict(csv_start=str(start), utc_minus_csv_h=-hours, **dict(zip(NAMES, nse)), mean=float(np.mean(nse))))
    pd.DataFrame(scan).to_csv(out / 'synxflow_clock_scan.csv', index=False)

    # SynxFlow, CST, frames 0-95 with offset on 0-95 (check value).
    o = obs['CST']
    sh = o.mean(0) - sim_wse.mean(0)
    checks['synxflow_CST_frames0_95_nse'] = [round(nash_sutcliffe_efficiency(sim_wse[:, i] + sh[i], o[:, i]), 3)
                                             for i in range(6)]
    (out / 'e1e2_checks.json').write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, default=P.ARRAYS)
    p.add_argument('--out', type=Path,
                   default=P.OUTPUTS / 'e1e2')
    p.add_argument('--device', default='cuda:0')
    main(p.parse_args())
