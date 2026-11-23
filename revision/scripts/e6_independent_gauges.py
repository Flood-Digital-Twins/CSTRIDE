"""E6: validation at basin USGS gauges that are never model inputs (April 2013, CST clock).

Gauges: every site in the notebook's h/ folder except the six inputs, with >= 80 % hourly
coverage on the CST window 2013-04-17 02:00 .. 04-21 01:00 (a raw sample within 30 min of the hour).
Cells per gauge:
  xs       notebook rule: 60 m cross-section perpendicular to the nearest NHD flowline_mr,
           first 2 rasterised cells; WSE = max depth + mean dem_5070
  nearest  the raster cell containing the gauge point; WSE = depth + dem
Sources at frames 12-95: SynxFlow (prepared truth), C-STRIDE full/terrain/vanilla/four_gauges
with simulated or USGS (CST) gauge inputs at the six released input cells.
Every series is centred on its own 12-95 mean (notebook convention).
"""
import paths as P  # noqa: E402  (input and output locations, see paths.py)
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e1_usgs_clock import CS  # noqa: E402
from e4_fixed_grid import fixed_grid_features  # noqa: E402

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
import torch
import yaml
from rasterio import features
from shapely.geometry import LineString

from cstride.chicago import (MinMaxNormalizer, build_sensor_sequences, load_chicago_manifest, load_chicago_sensors,
                             make_windows, chicago_include_precipitation_input, chicago_use_terrain_conditioning)
from cstride.chicago_forecast import build_chicago_feature_coords
from score_additional_metrics import encode_latent, load_state_dict_compatible
from cstride.models import CSTRIDE

G = str(P.USGS_2013)
OUT = P.OUTPUTS / 'e6'
E1 = P.OUTPUTS / 'e1e2'
DATA = P.ARRAYS
H = os.path.join(G, 'USGS_2013-04-15_2013-04-23/h')
FILES = sorted(os.listdir(H), key=lambda x: int(x.split('_')[0]))
INPUTS = {0, 7, 14, 19, 20, 25}
TIMES = pd.date_range('2013-04-17 02:00', periods=96, freq='h')  # CSV naive (CST) of frames 0..95
URL = 'https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/4/query'
MODELS = ['full', 'terrain', 'vanilla', 'four_gauges']


def usgs(g):
    d = pd.read_csv(os.path.join(H, FILES[g]), dtype={'site_no': str})
    if not len(d) or d.columns[1] != 'Gage Height (ft)':
        return None, 0.0, d
    t = pd.to_datetime(d['datetime'])
    s = pd.Series(pd.to_numeric(d['Gage Height (ft)'], errors='coerce').values, index=t).dropna()
    s = s[~s.index.duplicated(keep='last')].sort_index()
    cov = float(s.reindex(TIMES, method='nearest', tolerance=pd.Timedelta('30min')).notna().mean())
    w = s.loc[TIMES[0]:TIMES[-1]].resample('h').nearest().reindex(TIMES)
    w = w.interpolate(limit_direction='both')
    return (w.to_numpy() + float(d.alt_va.iloc[0])) * 0.3048, cov, d


def fetch_flowlines(pts):
    lon0, lat0, lon1, lat1 = pts.total_bounds
    params = dict(geometry=f'{lon0 - .03},{lat0 - .03},{lon1 + .03},{lat1 + .03}', geometryType='esriGeometryEnvelope',
                  inSR=4326, outSR=4326, spatialRel='esriSpatialRelIntersects', outFields='*', returnGeometry='true',
                  f='geojson', resultOffset=0, resultRecordCount=2000)
    feats = []
    while True:
        js = requests.get(URL, params=params, timeout=180).json()
        feats += js['features']
        if not js.get('exceededTransferLimit') and len(js['features']) < params['resultRecordCount']:
            break
        params['resultOffset'] += len(js['features'])
    return gpd.GeoDataFrame.from_features(feats, crs=4326).to_crs(5070)


def cross_at_point(flowline, point, half_width):
    d = flowline.project(point)
    nearest = flowline.interpolate(d)
    tangent = flowline.interpolate(d + 1)
    dx, dy = tangent.x - nearest.x, tangent.y - nearest.y
    L = np.hypot(dx, dy)
    nx, ny = -dy / L, dx / L
    return LineString([(nearest.x + nx * half_width, nearest.y + ny * half_width),
                       (nearest.x - nx * half_width, nearest.y - ny * half_width)])


def timing(series):
    lo, hi = series.min(), series.max()
    level = lo + 0.5 * (hi - lo)
    idx = np.flatnonzero(series >= level)
    w = series[idx] - level
    return int(idx[0]), float(np.sum(idx * w) / np.sum(w)) if w.sum() > 0 else float(idx.mean())


def score(sim, obs):
    if np.std(sim) < 1e-6:
        return dict(nse=np.nan, r=np.nan, alpha=0.0, rmse_m=float(np.sqrt(np.mean((sim - sim.mean() - obs + obs.mean()) ** 2))),
                    half_rise_time_h=np.nan, centroid_time_h=np.nan)
    s = sim - sim.mean() + obs.mean()
    hs, cs = timing(s)
    ho, co = timing(obs)
    return dict(nse=float(1 - np.sum((s - obs) ** 2) / np.sum((obs - obs.mean()) ** 2)),
                r=float(np.corrcoef(s, obs)[0, 1]), alpha=float(s.std() / obs.std()),
                rmse_m=float(np.sqrt(np.mean((s - obs) ** 2))), half_rise_time_h=hs - ho, centroid_time_h=round(cs - co, 2))


def main(device='cuda:0'):
    OUT.mkdir(parents=True, exist_ok=True)
    pts = gpd.read_file(os.path.join(G, 'gages.geojson'))
    pts['geometry'] = pts.geometry.apply(lambda g: g.geoms[0] if g.geom_type == 'MultiPoint' else g)
    pts = pts.set_crs(4326, allow_override=True)
    fl = fetch_flowlines(pts)
    fl.to_file(OUT / 'nhd_flowline_mr_all_gauges.gpkg', driver='GPKG')
    p5070 = pts.to_crs(5070)
    with rasterio.open(os.path.join(G, 'dem_5070.tif')) as src:
        dem = src.read(1)
        shape, transform, inv = (src.height, src.width), src.transform, ~src.transform
    truth = np.load(os.path.join(G, 'truth_traj120.npy'), mmap_mode='r')

    gauges, obs = [], {}
    for g in range(len(FILES)):
        if g in INPUTS:
            continue
        o, cov, d = usgs(g)
        info = dict(gage_index=g, site_no=FILES[g].split('_')[1], station_nm=d.station_nm.iloc[0] if len(d) else '',
                    coverage=cov)
        if o is None or cov < 0.8:
            info['excluded'] = 'no data' if o is None else f'coverage {cov:.2f}'
            gauges.append(info)
            continue
        pt = p5070.geometry.iloc[g]
        dist = fl.distance(pt)
        line = fl.geometry.loc[dist.idxmin()]
        m = features.rasterize([(cross_at_point(line, pt, 30.0), 1)], out_shape=shape, transform=transform,
                               all_touched=True, fill=0, dtype='uint8')
        rr, cc = np.where(m == 1)
        col, row = inv * (pt.x, pt.y)
        info.update(distance_to_flowline_m=float(dist.min()), flowline_name=str(fl.loc[dist.idxmin()].get('gnis_name')),
                    xs_cells=[[int(a), int(b)] for a, b in zip(rr[:2], cc[:2])], nearest_cell=[int(row), int(col)])
        for kind, cells in [('xs', info['xs_cells']), ('nearest', [info['nearest_cell']])]:
            r_, c_ = np.array([x[0] for x in cells]), np.array([x[1] for x in cells])
            h = np.asarray(truth[:, r_, c_, 0], dtype=np.float32)
            info[f'{kind}_dem_m'] = float(dem[r_, c_].mean())
            info[f'{kind}_dry_before_frame12'] = bool(h[:12].max() < 0.05)
            info[f'{kind}_max_depth_m'] = float(h.max())
        obs[g] = o
        gauges.append(info)
    (OUT / 'e6_gauges.json').write_text(json.dumps(gauges, indent=1))
    kept = [x for x in gauges if 'excluded' not in x]
    print('kept', len(kept), 'excluded', [(x['site_no'], x['excluded']) for x in gauges if 'excluded' in x], flush=True)

    # All query cells, in a fixed order.
    q = []
    for x in kept:
        for kind in ('xs', 'nearest'):
            cells = x['xs_cells'] if kind == 'xs' else [x['nearest_cell']]
            for c in cells:
                q.append((x['gage_index'], kind, c[0], c[1]))
    qr = np.array([a[2] for a in q])
    qc = np.array([a[3] for a in q])
    flat = qr * shape[1] + qc
    coords, elev, man, slo = fixed_grid_features(120, flat, shape)
    dem_q = dem[qr, qc].astype(np.float32)
    series = {}
    h_truth = np.asarray(truth[:, qr, qc, 0], dtype=np.float32)

    def to_wse(depth_by_cell):  # depth (T, n_query) -> {(g, kind): WSE(T)}
        out = {}
        for x in kept:
            for kind in ('xs', 'nearest'):
                sel = [i for i, a in enumerate(q) if a[0] == x['gage_index'] and a[1] == kind]
                out[x['gage_index'], kind] = depth_by_cell[:, sel].max(axis=1) + dem_q[sel].mean()
        return out

    series['SynxFlow'] = to_wse(h_truth)
    root, manifest, entries = load_chicago_manifest(DATA)
    zn = np.load(CS / 'data/normalization/normalizers.npz')
    norms = {k: MinMaxNormalizer(zn[k + '_min'], zn[k + '_max'])
             for k in ['state', 'depth', 'precip', 'elevation', 'manning', 'slope']}
    entry = next(e for e in entries if e['traj_id'] == 120)
    gidx = np.asarray(json.load(open(CS / 'data/gauges/USGS_6sensors.json'))['120'])
    inp = np.load(E1 / 'usgs_observations_CST_full.npz')['input_depth']
    cwd = os.getcwd()
    os.chdir(CS)
    for label in MODELS:
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
        feats = torch.as_tensor(build_chicago_feature_coords(coords, elev, man, slo, norms, terrain), device=device)[None]
        seq = build_sensor_sequences([entry], root, smap, norms, [0], include_precipitation=precip)[0]
        positions = [int(np.flatnonzero(gidx == i)[0]) for i in smap[120]]
        custom = seq.clone()
        custom[:, :len(positions)] = torch.tensor(norms['depth'].transform(inp[:, positions]))
        for vname, sq in [('sim', seq), ('USGS-CST', custom)]:
            res = []
            with torch.inference_mode():
                for w in make_windows(sq, lag):
                    latent = encode_latent(model, w[None].to(device))
                    res.append(norms['depth'].inverse_transform(model.decode(feats, latent).cpu().numpy().reshape(-1)))
            depth = np.concatenate([np.full((lag, len(q)), np.nan, np.float32), np.asarray(res)])
            series[f'C-STRIDE {label} ({vname} input)'] = to_wse(depth)
        del model
        torch.cuda.empty_cache()
    os.chdir(cwd)

    rows = []
    for src, d in series.items():
        for (g, kind), s in d.items():
            x = next(k for k in kept if k['gage_index'] == g)
            rows.append(dict(source=src, gage_index=g, site_no=x['site_no'], station_nm=x['station_nm'], cells=kind,
                             dry_before_frame12=x[f'{kind}_dry_before_frame12'],
                             synxflow_max_depth_m=x[f'{kind}_max_depth_m'], **score(s[12:96], obs[g][12:96])))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'e6_scores_long.csv', index=False)
    np.savez_compressed(OUT / 'e6_series.npz', **{f'{a}__{g}__{k}': v for a, d in series.items() for (g, k), v in d.items()},
                        **{f'USGS__{g}': o for g, o in obs.items()})


if __name__ == '__main__':
    main()
