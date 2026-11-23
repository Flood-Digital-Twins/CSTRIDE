"""E3 step 3: rebuild the notebook's cross-sections from NHD medium-resolution flowlines
(pynhd 'flowline_mr' = hydro.nationalmap.gov nhd/MapServer layer 4), with the notebook's
cross_at_point / rasterize / first-k logic, and check them against the exact-match cells."""
import paths as P  # noqa: E402  (input and output locations, see paths.py)
import json
import os
import geopandas as gpd
import numpy as np
import rasterio
import requests
from rasterio import features
from shapely.geometry import LineString

G = str(P.USGS_2013)
OUT = str(P.OUTPUTS / 'e3')
GAUGES = [0, 7, 14, 19, 20, 25]
W90 = [0, 19, 20, 25]
URL = 'https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/4/query'

pts = gpd.read_file(os.path.join(G, 'gages.geojson'))
pts['geometry'] = pts.geometry.apply(lambda g: g.geoms[0] if g.geom_type == 'MultiPoint' else g)
pts = pts.set_crs(4326, allow_override=True)
lon_min, lat_min, lon_max, lat_max = pts.iloc[GAUGES].total_bounds
pad = 0.03
params = dict(geometry=f'{lon_min - pad},{lat_min - pad},{lon_max + pad},{lat_max + pad}',
              geometryType='esriGeometryEnvelope', inSR=4326, outSR=4326, spatialRel='esriSpatialRelIntersects',
              outFields='*', returnGeometry='true', f='geojson', resultOffset=0, resultRecordCount=2000)
feats = []
while True:
    r = requests.get(URL, params=params, timeout=120)
    r.raise_for_status()
    js = r.json()
    feats += js['features']
    if not js.get('exceededTransferLimit') and len(js['features']) < params['resultRecordCount']:
        break
    params['resultOffset'] += len(js['features'])
fl = gpd.GeoDataFrame.from_features(feats, crs=4326).to_crs(5070)
fl.to_file(os.path.join(OUT, 'nhd_flowline_mr_subset.gpkg'), driver='GPKG')
print('flowlines fetched', len(fl))
pts5070 = pts.to_crs(5070)
with rasterio.open(os.path.join(G, 'dem_5070.tif')) as src:
    shape, transform = (src.height, src.width), src.transform


def cross_at_point(flowline, point, half_width):
    d = flowline.project(point)
    nearest = flowline.interpolate(d)
    tangent = flowline.interpolate(d + 1)
    dx, dy = tangent.x - nearest.x, tangent.y - nearest.y
    L = np.hypot(dx, dy)
    nx, ny = -dy / L, dx / L
    return LineString([(nearest.x + nx * half_width, nearest.y + ny * half_width),
                       (nearest.x - nx * half_width, nearest.y - ny * half_width)])


exact = json.load(open(os.path.join(OUT, 'e3_crosssection_cells.json')))
res = {}
for g in GAUGES:
    w = 90 if g in W90 else 60
    pt = pts5070.geometry.iloc[g]
    dist = fl.distance(pt)
    line = fl.geometry.loc[dist.idxmin()]
    cs = cross_at_point(line, pt, w / 2)
    lines = cs.geoms if cs.geom_type == 'MultiLineString' else [cs]
    cells = []
    for ln in lines:
        m = features.rasterize([(ln, 1)], out_shape=shape, transform=transform, all_touched=True, fill=0, dtype='uint8')
        rr, cc = np.where(m == 1)
        cells += list(zip(rr[:w // 30].tolist(), cc[:w // 30].tolist()))
    key = sorted([list(c) for c in cells])
    ok = key in [sorted(m) for m in exact[str(g)]['matches']]
    res[g] = dict(site_no=str(pts.staid.iloc[g]), width_m=w, distance_to_flowline_m=float(dist.min()),
                  comid=str(fl.loc[dist.idxmin()].get('comid', fl.loc[dist.idxmin()].get('COMID', ''))),
                  cells=[list(c) for c in cells], in_exact_match_set=ok)
    print(g, res[g])
json.dump(res, open(os.path.join(OUT, 'e3_crosssection_cells_nhd.json'), 'w'), indent=1)
