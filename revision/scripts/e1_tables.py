"""Compact Table-7-layout CSVs from e1e2_usgs_metrics_long.csv (run inside results/e1e2)."""
import pandas as pd
df = pd.read_csv('e1e2_usgs_metrics_long.csv', dtype={'station': str})
MODELS = ['full', 'terrain', 'vanilla', 'four_gauges']

def wide(d):
    w = d.pivot_table(index=['model', 'station', 'name', 'role', 'offset_m'], columns='source',
                      values=['nse', 'kge', 'peak_timing_hours'], sort=False)
    w.columns = [f'{s}_{m}' for m, s in w.columns]
    order = [f'{s}_{m}' for s in ['SynxFlow', 'sim_input', 'usgs_input'] for m in ['nse', 'kge', 'peak_timing_hours']
             if f'{s}_{m}' in w.columns]
    w = w[order].reset_index().round(3)
    for c in w.columns:
        if c.endswith('peak_timing_hours'):
            w[c] = w[c].astype('Int64')
    return w

for c in ['CST', 'CDT', 'published']:
    wide(df[(df.clock == c) & (df.input_offset == 'full') & df.model.isin(MODELS)]).to_csv(f'table7_{c}_offset12-95.csv', index=False)
for c in ['CST', 'CDT']:
    for so in ['causal', 'full']:
        wide(df[(df.clock == c) & (df.input_offset == 'causal') & (df.scoring_offset == so) & df.model.isin(MODELS)]
             ).to_csv(f'e2_{c}_inputcausal_score{so}.csv', index=False)
rows = []
for (c, io, so, m, src), g in df.groupby(['clock', 'input_offset', 'scoring_offset', 'model', 'source']):
    r = dict(clock=c, input_offset=io, scoring_offset=so, model=m, source=src, mean_nse_6=g.nse.mean(),
             mean_kge_6=g.kge.mean(), mean_abs_peak_h=g.peak_timing_hours.abs().mean())
    if m == 'four_gauges':
        r['mean_nse_input4'] = g[g.role == 'input'].nse.mean()
        r['mean_nse_heldout2'] = g[g.role == 'held_out'].nse.mean()
    rows.append(r)
pd.DataFrame(rows).round(3).to_csv('summary_means.csv', index=False)
