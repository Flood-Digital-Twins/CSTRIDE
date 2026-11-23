"""E5 data: the prepared Chicago arrays (CSTRIDE_ARRAYS) with the Manning channel corrected
(prepare_arrays.py without --legacy-manning-order). Every other file is a symlink to the original.

Per trajectory the event cells are recomputed as prepare_arrays.py does (finite DEM, finite depth on
frames 1..96, peak depth >= 0.1 m) and validated before writing: the cell count, the float16
coordinates and the LEGACY Manning at those cells must equal the original arrays exactly.
The DEM and manning.dat are identical for every event (md5 checked), so the rasters are built once.
"""
import paths as P  # noqa: E402  (input and output locations, see paths.py)
import importlib.util
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

CS = P.CSTRIDE_ROOT
SRC = P.ARRAYS
DST = P.OUTPUTS / 'e5' / 'data' / 'Chicago'
SIMS = P.SIMS
APRIL = SIMS / 'event_2013-04-10_2013-04-23_manning_water_0.02_manning_land_0.05'

spec = importlib.util.spec_from_file_location('prepare_arrays', CS / 'preprocessing/prepare_arrays/prepare_arrays.py')
prepare_arrays = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_arrays)


def event_dir(k):
    return APRIL if k == 120 else SIMS / f'event_{k}_filling_hours_0_filling_intensity_mm_hr_0'


def rasters():
    out = {}
    for legacy in (True, False):
        elev, valid, mean, std, gx, gy, manning = prepare_arrays.raw_static(event_dir(0), legacy)
        out[legacy] = (manning * 100).astype(np.float16).astype(np.float32) / 100
    return out, valid, elev.shape


def one(args):
    k, entry = args
    rough, valid0, shape = RASTERS
    height, width = shape
    flow = np.load(event_dir(k) / 'flow_variables.npy', mmap_mode='r')
    peak = np.full(shape, -np.inf, np.float32)
    valid = valid0.copy()
    for t in range(1, 97):
        d = np.asarray(flow[t, 0], dtype=np.float32)
        valid &= np.isfinite(d)
        peak = np.maximum(peak, d)
    valid &= peak >= 0.1
    cells = np.flatnonzero(valid.ravel())
    rows, cols = np.unravel_index(cells, shape)
    x = 2 * (np.arange(width, dtype=np.float32) / width - .5)
    y = 2 * (np.arange(height, dtype=np.float32) / height - .5) * (height / float(width))
    coords = np.column_stack([x[cols], y[rows]]).astype(np.float16)
    old_coords = np.load(SRC / entry['coords_file'])
    old_man = np.load(SRC / entry['manning_file'])
    rep = dict(traj_id=k, cells=int(len(cells)), original_cells=int(len(old_man)))
    rep['coords_equal'] = bool(len(cells) == len(old_man) and np.array_equal(coords, old_coords))
    legacy = rough[True].ravel()[cells]
    rep['legacy_manning_equal'] = bool(len(cells) == len(old_man) and np.array_equal(legacy.astype(old_man.dtype), old_man))
    if not (rep['coords_equal'] and rep['legacy_manning_equal']):
        return rep
    corrected = rough[False].ravel()[cells].astype(old_man.dtype)
    np.save(DST / entry['manning_file'], corrected)
    np.save(DST / f'cell_indices_traj{k}.npy', cells)
    rep['changed_cells'] = int(np.count_nonzero(corrected != old_man))
    rep['water_share_legacy'] = float(np.mean(old_man == 0.02))
    rep['water_share_corrected'] = float(np.mean(corrected == 0.02))
    return rep


RASTERS = None

if __name__ == '__main__':
    DST.mkdir(parents=True, exist_ok=True)
    RASTERS = rasters()
    manifest = json.loads((SRC / 'manifest.json').read_text())
    for name in os.listdir(SRC):
        if name.startswith('manning_traj') or name == 'manifest.json':
            continue
        link = DST / name
        if not link.exists():
            link.symlink_to(SRC / name)
    manifest['notes'] = manifest.get('notes', []) + [
        'E5 (2026-11-21): manning_traj*.npy recomputed with the corrected (non-legacy) Manning order; all other files are '
        f'symlinks to {P.ARRAYS}.']
    (DST / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    jobs = [(int(e['traj_id']), e) for e in manifest['files']]
    with Pool(8) as pool:
        reports = pool.map(one, jobs)
    (DST.parent / 'e5_data_report.json').write_text(json.dumps(reports, indent=1))
    bad = [r for r in reports if not (r['coords_equal'] and r['legacy_manning_equal'])]
    print('trajectories', len(reports), 'failed validation', len(bad), bad[:3])
    ok = [r for r in reports if r not in bad]
    print('mean water share legacy %.4f corrected %.4f; mean changed share %.4f' % (
        np.mean([r['water_share_legacy'] for r in ok]), np.mean([r['water_share_corrected'] for r in ok]),
        np.mean([r['changed_cells'] / r['cells'] for r in ok])))
    sys.exit(1 if bad else 0)
