"""E3 step 2: recover the notebook's cross-section cells for the six gauges.

The notebook rasterised a 60 or 90 m line perpendicular to the nearest NHD flowline and kept
the first width/30 cells (np.where row-major order), then stored
  wse(t) = max_cells h(t) + mean_cells dem.
NHD flowlines are not available here, so search all 2- or 3-cell sets inside 3x3 blocks near the
gauge for an exact match to the stored SynxFlow series. Only an exact, unique match is accepted.
"""
import paths as P  # noqa: E402  (input and output locations, see paths.py)
import itertools
import json
import os
import numpy as np
import geopandas as gpd
import rasterio
import xarray as xr

G = str(P.USGS_2013)
OUT = str(P.OUTPUTS / 'e3')
GAUGES = [0, 7, 14, 19, 20, 25]
W90 = [0, 19, 20, 25]
R = 12
truth = np.load(os.path.join(G, 'truth_traj120.npy'), mmap_mode='r')
with rasterio.open(os.path.join(G, 'dem_5070.tif')) as src:
    dem = src.read(1)
    inv = ~src.transform
pts = gpd.read_file(os.path.join(G, 'gages.geojson'))
pts['geometry'] = pts.geometry.apply(lambda g: g.geoms[0] if g.geom_type == 'MultiPoint' else g)
pts = pts.set_crs(4326, allow_override=True).to_crs(5070)
ds = {w: xr.open_dataset(os.path.join(G, f'output/cross_section_metrics_synxflow_width_{w}.nc')) for w in (60, 90)}
result = {}
for g in GAUGES:
    w = 90 if g in W90 else 60
    k = w // 30
    target = ds[w]['wse'].isel(gauge=g).values.astype(np.float64)
    col, row = inv * (pts.geometry.iloc[g].x, pts.geometry.iloc[g].y)
    r0, c0 = int(row), int(col)
    h = np.asarray(truth[:, r0 - R:r0 + R + 1, c0 - R:c0 + R + 1, 0], dtype=np.float32)
    dm = dem[r0 - R:r0 + R + 1, c0 - R:c0 + R + 1]
    matches = set()
    for br in range(0, 2 * R - 1):
        for bc in range(0, 2 * R - 1):
            cells = [(br + i, bc + j) for i in range(3) for j in range(3)]
            for sub in itertools.combinations(cells, k):
                rr = np.array([s[0] for s in sub]); cc = np.array([s[1] for s in sub])
                series = h[:, rr, cc].max(axis=1).astype(np.float64) + dm[rr, cc].astype(np.float32).mean(dtype=np.float32)
                if np.max(np.abs(series - target)) < 1e-4:
                    matches.add(tuple(sorted((int(r0 - R + a), int(c0 - R + b)) for a, b in sub)))
    # Discard sets whose extra cells are always dry and add nothing (they change mean dem, so they
    # cannot match unless dem is equal); report all matches.
    result[g] = dict(site_no=str(pts.staid.iloc[g]), width_m=w, n_cells=k, gauge_pixel=[r0, c0],
                     matches=[list(map(list, m)) for m in sorted(matches)])
    print(g, pts.staid.iloc[g], w, 'gauge pixel', (r0, c0), 'matches', len(matches), sorted(matches)[:4])
json.dump(result, open(os.path.join(OUT, 'e3_crosssection_cells.json'), 'w'), indent=1)
