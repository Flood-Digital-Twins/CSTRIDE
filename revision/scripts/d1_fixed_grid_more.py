"""D1: E4 protocol (CLDNet fixed 1,408,587-cell grid and each event's >= 0.1 m mask, frames 12-95,
0.5 m inundation threshold) for C-STRIDE vanilla and forcing, and for the saved CLDNet + LD-EnSF
predictions (the encoder weights are not available, so the saved fields are scored).
LD-EnSF fields: predictions_traj{107,108,109,120}.npz in LDENSF_PREDICTIONS (see paths.py),
(96, 5075, 1661, 3) float32.
"""
import paths as P  # noqa: E402  (input and output locations, see paths.py)
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e4_fixed_grid import (Acc, fixed_grid_features, CS, CLD, DATA, SIMS, EVENT_DIR, TS)  # noqa: E402

import numpy as np
import pandas as pd
import torch
import yaml

from cstride.chicago import (MinMaxNormalizer, build_sensor_sequences, load_chicago_manifest, load_chicago_sensors,
                             make_windows, chicago_include_precipitation_input, chicago_use_terrain_conditioning)
from cstride.chicago_forecast import build_chicago_feature_coords
from score_additional_metrics import encode_latent, load_state_dict_compatible
from cstride.models import CSTRIDE

OUT = P.OUTPUTS / 'd1'
LDENSF = {107: str(P.LDENSF / 'predictions_traj107.npz'),
          108: str(P.LDENSF / 'predictions_traj108.npz'),
          109: str(P.LDENSF / 'predictions_traj109.npz'),
          120: str(P.LDENSF / 'predictions_traj120.npz')}


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
    for label in ['vanilla', 'forcing']:
        path = CS / 'checkpoints' / label
        config = yaml.safe_load((path / 'config.yaml').read_text())
        precip = chicago_include_precipitation_input(config)
        terrain = chicago_use_terrain_conditioning(config)
        smap = load_chicago_sensors(path, config)
        model = CSTRIDE(input_size=len(smap[120]) + 507 * precip, coord_size=5 if terrain else 2, output_size=1,
                        device=device, **config['model']).to(device)
        model.load_state_dict(load_state_dict_compatible(path / 'best_model.ckpt', device))
        model.eval()
        models[label] = (model, smap, precip, terrain, int(config['basics']['lag']))
    rows, meta = [], {}
    for event in [107, 108, 109, 120]:
        entry = next(e for e in entries if e['traj_id'] == event)
        coords, elev, man, slo = fixed_grid_features(event, cells, shape)
        g = np.load(CS / f'data/grids/grid_traj{event}.npz')
        ev = g['cell_indices']
        in_event = np.zeros(len(cells), bool)
        in_event[pos[ev[pos[ev] >= 0]]] = True
        outside = ~in_event
        flow = np.load(SIMS / EVENT_DIR[event] / 'flow_variables.npy', mmap_mode='r')
        ld = np.load(LDENSF[event])
        meta[event] = dict(traj_id=int(ld['traj_id']), background_traj_id=int(ld['background_traj_id']),
                           da_seed=int(ld['da_seed']), obs_idx=ld['obs_idx'].tolist())
        ld_h = np.ascontiguousarray(ld['pred_full'][..., 0].reshape(96, -1)[:, cells])
        acc = {(m, s): Acc() for m in ['C-STRIDE vanilla', 'C-STRIDE forcing', 'CLDNet + LD-EnSF']
               for s in ['fixed_grid', 'event_mask']}
        feats, wins = {}, {}
        for label, (model, smap, precip, terrain, lag) in models.items():
            feats[label] = torch.as_tensor(build_chicago_feature_coords(coords, elev, man, slo, norms, terrain), device=device)
            seq = build_sensor_sequences([entry], root, smap, norms, [0], include_precipitation=precip)[0]
            wins[label] = make_windows(seq, lag)
        for t in TS:
            truth = np.asarray(flow[t + 1, 0], dtype=np.float32).ravel()[cells]
            preds = {'CLDNet + LD-EnSF': ld_h[t]}
            with torch.inference_mode():
                for label, (model, smap, precip, terrain, lag) in models.items():
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
        del ld, ld_h
        print(event, meta[event], flush=True)
        pd.DataFrame(rows).to_csv(OUT / 'd1_fixed_grid_long.csv', index=False)
    (OUT / 'd1_ldensf_meta.json').write_text(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
