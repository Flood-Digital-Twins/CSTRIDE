"""D2: timing, r and alpha at the Si et al. cross-sections (CST, frames 12-95, own-mean centring).

Reads results/e3/e3_crosssection_scores_long.csv (written by e3_crosssection_scores.py) and writes
results/d2/: the long table, a per-gauge timing table (half-rise and centroid, series minus USGS, in
hours) and means over the six gauges. In four_gauges, Salt Creek and DuPage are held-out inputs.
"""
import paths as P  # noqa: E402  (input and output locations, see paths.py)
import pandas as pd

SRCS = {'SynxFlow': 'SynxFlow', 'C-STRIDE full (sim input)': 'full_sim', 'C-STRIDE full (USGS-CST input)': 'full_usgs',
        'C-STRIDE four_gauges (sim input)': '4g_sim', 'C-STRIDE four_gauges (USGS-CST input)': '4g_usgs'}
HELD = {'05531500', '05540130'}


def main():
    out = P.OUTPUTS / 'd2'
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(P.OUTPUTS / 'e3' / 'e3_crosssection_scores_long.csv', dtype={'station': str})
    d = df[(df.clock == 'CST') & (df.frames == '12-95') & df.source.isin(SRCS)].copy()
    d['src'] = d.source.map(SRCS)
    d['role'] = [('held_out' if (s in HELD and r.startswith('4g')) else 'input') for s, r in zip(d.station, d.src)]
    d[['src', 'station', 'name', 'role', 'nse', 'r', 'alpha', 'peak_time_h', 'half_rise_time_h',
       'centroid_time_h']].round(3).to_csv(out / 'd2_crosssection_timing_long.csv', index=False)
    w = d.pivot_table(index=['station', 'name'], columns='src', values=['half_rise_time_h', 'centroid_time_h'], sort=False)
    w = w.reindex(columns=[(m, s) for m in ['half_rise_time_h', 'centroid_time_h'] for s in SRCS.values()])
    w.columns = [f"{s}_{'half' if m.startswith('half') else 'cen'}" for m, s in w.columns]
    w.round(1).to_csv(out / 'd2_timing_table.csv')
    m = d.groupby('src', sort=False).agg(mean_r=('r', 'mean'), mean_alpha=('alpha', 'mean'), mean_nse=('nse', 'mean'),
                                         mean_abs_half=('half_rise_time_h', lambda x: x.abs().mean()),
                                         mean_abs_cen=('centroid_time_h', lambda x: x.abs().mean()))
    m.round(3).to_csv(out / 'd2_means.csv')
    print(w.round(1).to_csv())
    print(m.round(3).to_csv())


if __name__ == '__main__':
    main()
