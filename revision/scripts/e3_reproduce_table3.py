"""E3 step 1: reproduce the CLDNet notebook's SynxFlow-vs-USGS metrics (Si et al. Table 3) from
its saved cross-section WSE (.nc), and find which USGS clock it used.

The local CSVs are naive CST (UTC-6). The notebook first fetched NWIS IV and converted to
America/Chicago (CDT in April); CDT naive = CSV naive + 1 h.
"""
import paths as P  # noqa: E402  (input and output locations, see paths.py)
import os
import numpy as np
import pandas as pd
import xarray as xr

G = str(P.USGS_2013)
OUT = str(P.OUTPUTS / 'e3')
H = os.path.join(G, 'USGS_2013-04-15_2013-04-23/h')
files = sorted(os.listdir(H), key=lambda x: int(x.split('_')[0]))
GAUGES = [0, 7, 14, 19, 20, 25]
W90 = [0, 19, 20, 25]
ds = {w: xr.open_dataset(os.path.join(G, f'output/cross_section_metrics_synxflow_width_{w}.nc')) for w in (60, 90)}
start, end = pd.Timestamp('2013-04-17 08:00'), pd.Timestamp('2013-04-21 07:00')


def usgs(gidx, csv_to_label_hours, lo, hi):
    """Hourly USGS WSE with timestamps relabelled by +csv_to_label_hours, restricted to [lo, hi]."""
    d = pd.read_csv(os.path.join(H, files[gidx]))
    d['datetime'] = pd.to_datetime(d['datetime']) + pd.Timedelta(hours=csv_to_label_hours)
    d['Gage Height (ft)'] = pd.to_numeric(d['Gage Height (ft)'], errors='coerce')
    d = d.dropna(subset=['Gage Height (ft)']).sort_values('datetime')
    d = d[(d.datetime >= lo) & (d.datetime <= hi)]
    v = d[['datetime', 'Gage Height (ft)']].drop_duplicates('datetime', keep='last').set_index('datetime')
    v = v.resample('1h').nearest().loc[lo:hi]
    alt = (float(d.alt_va.iloc[0]) + (10 if gidx in (14, 20) else 0)) * 0.3048
    return (v['Gage Height (ft)'] * 0.3048 + alt).rename('wse')


def nse(o, s):
    return 1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)


def kge(o, s):
    r = np.corrcoef(s, o)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2)


def score(sim_shift_hours, csv_to_label_hours, lo, hi, frames=slice(0, 96)):
    """sim frame i labelled start + i + sim_shift_hours; USGS labelled CSV + csv_to_label_hours."""
    rows = []
    for g in GAUGES:
        x = ds[90 if g in W90 else 60]
        sim = pd.Series(x['wse'].isel(gauge=g).values[frames],
                        index=pd.to_datetime(x.time.values[frames]) + pd.Timedelta(hours=sim_shift_hours))
        o = usgs(g, csv_to_label_hours, lo, hi)
        sim = sim - sim.mean() + o.mean()
        j = pd.concat([sim.rename('sim'), o], axis=1, join='inner').dropna()
        rows.append(dict(gage_index=g, site_no=files[g].split('_')[1], n=len(j), nse=nse(j.wse.values, j.sim.values),
                         kge=kge(j.wse.values, j.sim.values),
                         r=np.corrcoef(j.sim, j.wse)[0, 1], alpha=j.sim.std(ddof=0) / j.wse.std(ddof=0),
                         abs_peak_time_h=abs((sim.idxmax() - o.idxmax()).total_seconds() / 3600)))
    return pd.DataFrame(rows)


pub = pd.read_csv(os.path.join(G, 'output/gauge_discharge_metrics.csv'))
pub = pub[pub.model == 'SynxFlow'].reset_index(drop=True)
cands = {
    'CSV-as-is (CST) [08,07]': (0, 0, start, end),
    'CDT [08,07]': (0, 1, start, end),
    'CDT [09,07] (CST-interpreted request)': (0, 1, start + pd.Timedelta(hours=1), end),
}
for name, (ss, ch, lo, hi) in cands.items():
    r = score(ss, ch, lo, hi)
    print(f'{name:40s} n={r.n.tolist()} max|dNSE|={np.abs(r.nse - pub.nse).max():.2e} '
          f'max|dKGE|={np.abs(r.kge - pub.kge).max():.2e} peak_dt={r.abs_peak_time_h.tolist()} vs {pub.time_to_peak_error_hours.tolist()}')

# Aligned: sim frame i is UTC start + i; label sim in CST naive (UTC-6) and use the CSV as is.
aligned = score(-6, 0, start - pd.Timedelta(hours=6), end - pd.Timedelta(hours=6))
published = score(0, 1, start + pd.Timedelta(hours=1), end)
out = published[['gage_index', 'site_no']].copy()
for k in ['n', 'nse', 'r', 'alpha', 'abs_peak_time_h']:
    out[f'notebook_{k}'] = published[k]
for k in ['n', 'nse', 'r', 'alpha', 'abs_peak_time_h']:
    out[f'aligned_CST_{k}'] = aligned[k]
out['published_table3_nse'] = pub.nse.values
out.round(3).to_csv(os.path.join(OUT, 'e3_synxflow_crosssection_vs_usgs.csv'), index=False)
print(out.round(3).to_string(index=False))
print('mean NSE notebook %.3f aligned %.3f' % (published.nse.mean(), aligned.nse.mean()))
