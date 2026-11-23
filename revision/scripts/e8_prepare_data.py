"""E8 data: all 94 events (90 'train' ids + 4 test) on CLDNet's fixed 1,408,587-cell grid
(grid.npz aggregate_mask, row-major), corrected Manning, clean split, one gauge cell per gauge.

prepare_arrays.py conventions: depth targets = flow_variables[1..96, depth] (one burn-in frame),
rainfall rows 0-95 of rain_source.npy (float16), coordinates by the C-STRIDE formula (float16,
equal to CLDNet's stored coords), elevation/slope with the released precision (quantized for
numbered events, raw float32 for 120), Manning corrected (no --legacy-manning-order).
Targets are stored depth-only (depth_file, (96, N, 1) float16): the loader reads depth_file when
no state_file is given, and training uses only the depth channel (targets and gauge histories).
Writes only under results/e8/ (REVISION_OUTPUTS).
"""
import paths as P  # noqa: E402  (input and output locations, see paths.py)
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e5_score import raster_features  # noqa: E402  (prepare_arrays rule on raster cells)

CS = P.CSTRIDE_ROOT
CLD = P.CLDNET
SRC = P.ARRAYS
SIMS = P.SIMS
APRIL = SIMS / 'event_2013-04-10_2013-04-23_manning_water_0.02_manning_land_0.05'
E8 = P.OUTPUTS / 'e8'
DST = E8 / 'data/Chicago'
GAUGE_RC = [(784, 781), (1324, 794), (2287, 1008), (3241, 1062), (3236, 1281), (3686, 460)]
STATIONS = ['05527800', '05528000', '05529000', '05531500', '05532500', '05540130']
SHAPE = (5075, 1661)


def event_dir(k):
    return APRIL if k == 120 else SIMS / f'event_{k}_filling_hours_0_filling_intensity_mm_hr_0'


CELLS = np.flatnonzero(np.load(CLD / 'data/illinois_grid/grid.npz')['aggregate_mask'])


def one(k):
    t0 = time.time()
    flow = np.load(event_dir(k) / 'flow_variables.npy', mmap_mode='r')
    assert flow.shape[1] == 3 and flow.shape[0] >= 97
    out = np.lib.format.open_memmap(DST / f'depth_traj{k}.npy', mode='w+', dtype=np.float16, shape=(96, len(CELLS), 1))
    nonfinite = 0
    for i, t in enumerate(range(1, 97)):
        d = np.asarray(flow[t, 0], dtype=np.float32).ravel()[CELLS]
        nonfinite += int((~np.isfinite(d)).sum())
        out[i, :, 0] = d.astype(np.float16)
    out.flush()
    del out
    rain = np.load(event_dir(k) / 'rain_source.npy', mmap_mode='r')
    if rain.ndim >= 3 and rain.shape[0] == 1:
        rain = rain[0]
    rain = np.asarray(rain[:96], dtype=np.float32).reshape(96, -1).astype(np.float16)
    np.save(DST / f'precip_traj{k}.npy', rain)
    ref = SRC / f'precip_traj{k}.npy'
    precip_equal = bool(np.array_equal(np.load(ref), rain)) if ref.exists() else None
    return dict(traj_id=k, seconds=round(time.time() - t0, 1), nonfinite_depth=nonfinite, precip_equal_released=precip_equal)


if __name__ == '__main__':
    t_start = time.time()
    DST.mkdir(parents=True, exist_ok=True)
    split = json.loads((CS / 'splits/chicago_split.json').read_text())
    train, test = sorted(split['train']), sorted(split['test'])
    ids = train + test
    assert len(train) == 90 and len(set(ids)) == 94

    # Static features (shared by all events; two precisions for elevation/slope).
    static = {}
    for tag, ev in (('numbered', 107), ('april', 120)):
        coords, elev, man, slo = raster_features(CELLS, SHAPE, ev, False)
        static[tag] = (coords, elev, man, slo)
    np.save(DST / 'coords_fixed.npy', static['numbered'][0])
    np.save(DST / 'manning_fixed_corrected.npy', static['numbered'][2])
    for tag in ('numbered', 'april'):
        np.save(DST / f'elevation_fixed_{tag}.npy', static[tag][1])
        np.save(DST / f'slope_fixed_{tag}.npy', static[tag][3])
    grid_coords = np.load(CLD / 'data/illinois_grid/grid.npz')['coords']
    checks = dict(cells=int(len(CELLS)), coords_equal_cldnet=bool(np.array_equal(grid_coords, static['numbered'][0])),
                  water_share_corrected=float(np.mean(static['numbered'][2] == 0.02)))

    # Gauges: one fixed-grid index per gauge, same for every event.
    pos = {int(c): i for i, c in enumerate(CELLS)}
    flat = [r * SHAPE[1] + c for r, c in GAUGE_RC]
    missing = [rc for rc, f in zip(GAUGE_RC, flat) if f not in pos]
    assert not missing, missing
    gidx = [pos[f] for f in flat]
    sensor_map = {str(k): gidx for k in ids}
    (E8 / 'sensor_indices_fixed.json').write_text(json.dumps(sensor_map))
    checks['gauges'] = dict(zip(STATIONS, [dict(row=r, col=c, fixed_index=i) for (r, c), i in zip(GAUGE_RC, gidx)]))

    # Clean split: 90 training events, no duplicates. The trainer needs a monitor list for its
    # validation loader; it only logs that loss here (no early stopping, final checkpoint used),
    # so it is set to one training event.
    e8_split = dict(dataset='E8 fixed grid', train=train, test=test, train_files=train,
                    checkpoint_monitor_files=[train[0]], not_used=[], checkpoint_selection='final_model.ckpt',
                    note='monitor list = one training event, logged only; no early stopping; final checkpoint')
    (E8 / 'split_e8.json').write_text(json.dumps(e8_split, indent=1))

    with Pool(8) as pool:
        reports = pool.map(one, ids)
    files = []
    for k in ids:
        tag = 'april' if k == 120 else 'numbered'
        files.append(dict(traj_id=k, depth_file=f'depth_traj{k}.npy', coords_file='coords_fixed.npy',
                          precip_file=f'precip_traj{k}.npy', elevation_file=f'elevation_fixed_{tag}.npy',
                          manning_file='manning_fixed_corrected.npy', slope_file=f'slope_fixed_{tag}.npy'))
    (DST / 'manifest.json').write_text(json.dumps(dict(
        time_steps=96, files=files, notes=['E8: CLDNet fixed grid (1,408,587 cells), corrected Manning, depth-only targets, '
                                           'burn-in 1, rainfall rows 0-95; written by scripts/e8_prepare_data.py']), indent=1))
    checks['reports'] = reports
    checks['wall_seconds'] = round(time.time() - t_start, 1)
    (E8 / 'e8_data_report.json').write_text(json.dumps(checks, indent=1))
    bad = [r for r in reports if r['nonfinite_depth'] or r['precip_equal_released'] is False]
    print('events', len(reports), 'wall %.0f s' % checks['wall_seconds'], 'coords==CLDNet', checks['coords_equal_cldnet'],
          'water share %.4f' % checks['water_share_corrected'], 'problems', bad[:5])
