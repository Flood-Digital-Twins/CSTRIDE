"""E7: diagnostics for E6 (inference only; not headline numbers).

E7a  E6 (both cell rules, CST, frames 12-95, own-mean centring, 19/20 unregulated gauges) and
     Table 7 (released sensor cells, SynxFlow offset on 12-95) for the robustness checkpoints
     noise005, noise010, noise020, dropout050 and, for reference, full; sim vs USGS-CST input.
E7b  RETROSPECTIVE UPPER BOUND. Full model with USGS stage mapped to input depth by a per-gauge
     linear map (scale and offset) that matches SynxFlow's mean and std at the sensor cell on
     frames 12-95; scored as E6 and Table 7.
E7c  From the E6 outputs (no inference): per gauge dNSE (USGS minus sim) for full and four_gauges
     vs distance to the nearest input gauge, which one, and NHDPlus up/downstream relation.
"""
import paths as P  # noqa: E402  (input and output locations, see paths.py)
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e1_usgs_clock import CS, STATIONS, NAMES  # noqa: E402
from e4_fixed_grid import fixed_grid_features  # noqa: E402
from e6_independent_gauges import score, G  # noqa: E402

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
import yaml

from cstride.chicago import (MinMaxNormalizer, build_sensor_sequences, load_chicago_manifest, load_chicago_sensors,
                             make_windows, chicago_include_precipitation_input, chicago_use_terrain_conditioning)
from cstride.chicago_forecast import build_chicago_feature_coords
from score_additional_metrics import encode_latent, load_state_dict_compatible, nash_sutcliffe_efficiency
from cstride.models import CSTRIDE

OUT = P.OUTPUTS / 'e7'
E1 = P.OUTPUTS / 'e1e2'
E6 = P.OUTPUTS / 'e6'
DATA = P.ARRAYS
REGULATED = {'05536995', '05537980'}
INPUT_GAUGE_INDEX = {'05527800': 0, '05528000': 19, '05529000': 20, '05531500': 7, '05532500': 25, '05540130': 14}


def inference(device='cuda:0'):
    gauges = [x for x in json.loads((E6 / 'e6_gauges.json').read_text()) if 'excluded' not in x]
    ser = np.load(E6 / 'e6_series.npz')
    q = [(x['gage_index'], kind, c[0], c[1]) for x in gauges for kind in ('xs', 'nearest')
         for c in (x['xs_cells'] if kind == 'xs' else [x['nearest_cell']])]
    with rasterio.open(os.path.join(G, 'dem_5070.tif')) as src:
        dem = src.read(1)
        shape = (src.height, src.width)
    qr, qc = np.array([a[2] for a in q]), np.array([a[3] for a in q])
    coords, elev, man, slo = fixed_grid_features(120, qr * shape[1] + qc, shape)
    dem_q = dem[qr, qc].astype(np.float32)
    sel = {(x['gage_index'], k): [i for i, a in enumerate(q) if a[0] == x['gage_index'] and a[1] == k]
           for x in gauges for k in ('xs', 'nearest')}

    root, manifest, entries = load_chicago_manifest(DATA)
    zn = np.load(CS / 'data/normalization/normalizers.npz')
    norms = {k: MinMaxNormalizer(zn[k + '_min'], zn[k + '_max'])
             for k in ['state', 'depth', 'precip', 'elevation', 'manning', 'slope']}
    entry = next(e for e in entries if e['traj_id'] == 120)
    gidx = np.asarray(json.load(open(CS / 'data/gauges/USGS_6sensors.json'))['120'])
    z = np.load(E1 / 'usgs_observations_CST_full.npz')
    obs6, sim_depth, el6, shift = z['obs_wse'], z['sim_depth'], z['elevation'], z['shift_m']
    sim_wse6 = sim_depth + el6
    a = sim_wse6[12:].std(0) / obs6[12:].std(0)
    b = sim_wse6[12:].mean(0) - a * obs6[12:].mean(0)
    inputs = {'USGS-CST': z['input_depth'], 'USGS-CST-linear': np.maximum(a * obs6 + b - el6, 0)}
    (OUT / 'e7b_linear_map.json').write_text(json.dumps(dict(station=STATIONS, scale=a.tolist(), offset_m=b.tolist()), indent=1))

    runs = [(m, v) for m in ['full', 'noise005', 'noise010', 'noise020', 'dropout050'] for v in ['sim', 'USGS-CST']]
    runs.append(('full', 'USGS-CST-linear'))
    rows6, rows7 = [], []
    cwd = os.getcwd()
    os.chdir(CS)
    for label in dict.fromkeys(m for m, _ in runs):
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
        all_coords = build_chicago_feature_coords(coords, elev, man, slo, norms, terrain)
        sensor_coords = build_chicago_feature_coords(*[np.load(root / entry[k + '_file'])[gidx] for k in
                                                       ('coords', 'elevation', 'manning', 'slope')], norms, terrain)
        feats = torch.as_tensor(np.concatenate([all_coords, sensor_coords]), device=device)[None]
        seq = build_sensor_sequences([entry], root, smap, norms, [0], include_precipitation=precip)[0]
        positions = [int(np.flatnonzero(gidx == i)[0]) for i in smap[120]]
        assert len(positions) == 6
        for m, v in runs:
            if m != label:
                continue
            sq = seq.clone()
            if v != 'sim':
                sq[:, :6] = torch.tensor(norms['depth'].transform(inputs[v][:, positions]))
            res = []
            with torch.inference_mode():
                for w in make_windows(sq, lag):
                    latent = encode_latent(model, w[None].to(device))
                    res.append(norms['depth'].inverse_transform(model.decode(feats, latent).cpu().numpy().reshape(-1)))
            pred = np.asarray(res)  # (84, nq + 6)
            src = f'{m} ({v} input)'
            for x in gauges:
                for k in ('xs', 'nearest'):
                    s = pred[:, sel[x['gage_index'], k]].max(axis=1) + dem_q[sel[x['gage_index'], k]].mean()
                    rows6.append(dict(source=src, model=m, input=v, site_no=x['site_no'], station_nm=x['station_nm'], cells=k,
                                      **score(s, ser[f"USGS__{x['gage_index']}"][12:96])))
            for i, st in enumerate(STATIONS):
                s = pred[:, len(q) + i] + el6[i] + shift[i]
                rows7.append(dict(source=src, model=m, input=v, station=st, name=NAMES[i],
                                  nse=nash_sutcliffe_efficiency(s, obs6[12:, i]), r=float(np.corrcoef(s, obs6[12:, i])[0, 1]),
                                  alpha=float(s.std() / obs6[12:, i].std())))
        del model
        torch.cuda.empty_cache()
    os.chdir(cwd)
    d6, d7 = pd.DataFrame(rows6), pd.DataFrame(rows7)
    d6['valid'] = ~((d6.site_no == '05530000') & (d6.cells == 'xs')) & ~d6.site_no.isin(REGULATED)
    d6.to_csv(OUT / 'e7ab_e6_scores_long.csv', index=False)
    d7.to_csv(OUT / 'e7ab_table7_long.csv', index=False)
    return d6, d7


def summarize(d6, d7):
    out = []
    for kind in ('xs', 'nearest'):
        d = d6[(d6.cells == kind) & d6.valid]
        for src, g in d.groupby('source', sort=False):
            out.append(dict(test='E6', cells=kind, source=src, n=len(g), mean_nse=g.nse.mean(), median_nse=g.nse.median(),
                            mean_r=g.r.mean(), mean_alpha=g.alpha.mean()))
        t = d.pivot_table(index='site_no', columns='source', values='nse')
        for m in ['full', 'noise005', 'noise010', 'noise020', 'dropout050']:
            for v in (['USGS-CST', 'USGS-CST-linear'] if m == 'full' else ['USGS-CST']):
                diff = t[f'{m} ({v} input)'] - t[f'{m} (sim input)']
                out.append(dict(test='E6', cells=kind, source=f'COUNT {m} {v} > sim', n=len(t), mean_nse=float((diff > 0).sum()),
                                median_nse=float(diff.median())))
    for src, g in d7.groupby('source', sort=False):
        out.append(dict(test='Table7', cells='sensor', source=src, n=len(g), mean_nse=g.nse.mean(), median_nse=g.nse.median(),
                        mean_r=g.r.mean(), mean_alpha=g.alpha.mean()))
    s = pd.DataFrame(out)
    s.round(3).to_csv(OUT / 'e7ab_summary.csv', index=False)
    return s


def e7c():
    gauges = {x['site_no']: x for x in json.loads((E6 / 'e6_gauges.json').read_text())}
    d = pd.read_csv(E6 / 'e6_scores_long.csv', dtype={'site_no': str})
    pts = gpd.read_file(os.path.join(G, 'gages.geojson'))
    pts['geometry'] = pts.geometry.apply(lambda g: g.geoms[0] if g.geom_type == 'MultiPoint' else g)
    pts = pts.set_crs(4326, allow_override=True).to_crs(5070)
    pts['staid'] = pts.staid.astype(str)
    fl = gpd.read_file(E6 / 'nhd_flowline_mr_all_gauges.gpkg')
    dn = dict(zip(fl.Hydroseq, fl.DnHydroseq))

    def hydroseq(site):
        p = pts.geometry[pts.staid == site].iloc[0]
        dist = fl.distance(p)
        return fl.Hydroseq.loc[dist.idxmin()], float(dist.min())

    def downstream(h):
        path, seen = [], set()
        while h in dn and h not in seen and h != 0:
            seen.add(h)
            path.append(h)
            h = dn[h]
        return set(path)

    inp = {s: hydroseq(s) for s in INPUT_GAUGE_INDEX}
    rows = []
    for site in d.site_no.unique():
        p = pts.geometry[pts.staid == site].iloc[0]
        dists = {s: p.distance(pts.geometry[pts.staid == s].iloc[0]) / 1000 for s in INPUT_GAUGE_INDEX}
        near = min(dists, key=dists.get)
        h, hd = hydroseq(site)
        rel = []
        for s, (hs, _) in inp.items():
            if hs in downstream(h):
                rel.append(f'upstream of {NAMES[STATIONS.index(s)]}')
            elif h in downstream(hs):
                rel.append(f'downstream of {NAMES[STATIONS.index(s)]}')
        r = dict(site_no=site, station_nm=gauges[site]['station_nm'], regulated=site in REGULATED,
                 nearest_input=NAMES[STATIONS.index(near)], distance_km=round(dists[near], 1),
                 nhd_distance_m=round(hd, 1), network_relation='; '.join(rel) or 'none (separate tributary)')
        for kind in ('xs', 'nearest'):
            for m in ('full', 'four_gauges'):
                x = d[(d.site_no == site) & (d.cells == kind)]
                if kind == 'xs' and site == '05530000':
                    r[f'{kind}_{m}_dNSE'] = np.nan
                    continue
                r[f'{kind}_{m}_dNSE'] = float(x[x.source == f'C-STRIDE {m} (USGS-CST input)'].nse.iloc[0]
                                              - x[x.source == f'C-STRIDE {m} (sim input)'].nse.iloc[0])
        rows.append(r)
    t = pd.DataFrame(rows).sort_values('distance_km')
    t.round(3).to_csv(OUT / 'e7c_dnse_vs_inputs.csv', index=False)
    return t


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    t = e7c()
    print(t.round(3).to_string(index=False))
    d6, d7 = inference()
    print(summarize(d6, d7).round(3).to_string(index=False))
