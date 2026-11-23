"""Derived tables: rebuilds every summary table in results/ from the long tables the experiment scripts write.

Run after the experiment scripts (see revision/README.md for the order). Writes:
  t1/   t1_r_alpha_table.csv, t1_timing_table.csv              (from t1_timing_long.csv)
        persistence_1h.csv                                    (1 h persistence of the USGS series)
  e2b/  e2b_split_offset_long.csv gains score_window, obs_std_m and rmse_m; e2b_nse_table.csv, e2b_rmse_table.csv
  e3/   e3_nse_table.csv                                      (NSE per gauge; last row = mean)
  e4/   e4_summary.csv                                        (CSI etc. pooled over the four events)
  d1/   d1_tables5_8_summary.csv                              (six models; CSI etc. for April 2013)
        fixed_grid_false_alarms.csv                           (six models, outside the event masks)
  e6/   e6_summary.csv, e6_nse_table.csv
  retraining_summary.csv                                      (released vs E5 vs E8, from the scores_* summaries)
"""
import paths as P  # noqa: E402  (input and output locations, see paths.py)
import json

import numpy as np
import pandas as pd

from summarize_scores import summarize

O = P.OUTPUTS
NAMES = ['Russell', 'Gurnee', 'Des Plaines', 'Salt Creek', 'Riverside', 'DuPage']
STATIONS = ['05527800', '05528000', '05529000', '05531500', '05532500', '05540130']
REGULATED = {'05536995', '05537980'}  # ship canal; Des Plaines at Joliet, below the canal confluence


def t1_tables():
    t = pd.read_csv(O / 't1/t1_timing_long.csv')
    t.pivot_table(index=['clock', 'model', 'source'], columns='name', values=['r', 'alpha'],
                  sort=False).round(3).to_csv(O / 't1/t1_r_alpha_table.csv')
    t.pivot_table(index=['clock', 'model', 'source'], columns='name', values=['half_rise_time_h', 'centroid_time_h'],
                  sort=False).round(1).to_csv(O / 't1/t1_timing_table.csv')


def persistence():
    rows = []
    for c in ['CST', 'CDT', 'published']:
        o = np.load(O / f'e1e2/usgs_observations_{c}_full.npz')['obs_wse']
        for i in range(6):
            ob, pr = o[12:96, i], o[11:95, i]
            rows.append(dict(clock=c, model='persistence_1h', source='usgs_obs_t-1', station=STATIONS[i], name=NAMES[i],
                             nse=1 - np.sum((pr - ob) ** 2) / np.sum((ob - ob.mean()) ** 2),
                             rmse_m=np.sqrt(np.mean((pr - ob) ** 2)), r=np.corrcoef(pr, ob)[0, 1], alpha=pr.std() / ob.std()))
    pd.DataFrame(rows).round(4).to_csv(O / 't1/persistence_1h.csv', index=False)


def e2b_tables():
    o = np.load(O / 'e1e2/usgs_observations_CST_full.npz')['obs_wse']
    sd = {'12-47': o[12:48].std(0), '48-95': o[48:96].std(0), '12-95': o[12:96].std(0)}
    f = O / 'e2b/e2b_split_offset_long.csv'
    e = pd.read_csv(f, dtype={'station': str})
    e['score_window'] = e.split.str.split('_score').str[1]
    e['obs_std_m'] = [sd[w][NAMES.index(n)] for w, n in zip(e.score_window, e.name)]
    e['rmse_m'] = np.sqrt((1 - e.nse)) * e.obs_std_m
    e.to_csv(f, index=False)
    e.pivot_table(index=['model', 'source', 'split'], columns='name', values='rmse_m', sort=False)[NAMES].round(3).to_csv(
        O / 'e2b/e2b_rmse_table.csv')
    e.pivot_table(index=['model', 'source', 'split'], columns='name', values='nse', sort=False)[NAMES].round(3).to_csv(
        O / 'e2b/e2b_nse_table.csv')


def e3_table():
    df = pd.read_csv(O / 'e3/e3_crosssection_scores_long.csv', dtype={'station': str})
    short = {'SynxFlow': 'SynxFlow', 'CLDNet': 'CLDNet', 'LDNet': 'LDNet'}
    for m, k in [('full', 'full'), ('terrain', 'terrain'), ('vanilla', 'vanilla'), ('four_gauges', '4g')]:
        short[f'C-STRIDE {m} (sim input)'] = f'{k}_sim'
        short[f'C-STRIDE {m} (USGS-CST input)'] = f'{k}_usgs'
    cols = ([('notebook', 'all', s) for s in ['SynxFlow', 'CLDNet', 'LDNet']]
            + [('CST', 'all', s) for s in ['SynxFlow', 'CLDNet', 'LDNet']] + [('CDT', 'all', 'SynxFlow')]
            + [('CST', '12-95', s) for s in ['SynxFlow', 'CLDNet']]
            + [('CST', '12-95', s) for s in short if s.startswith('C-STRIDE')])
    out = []
    for st, g in df.groupby('station', sort=False):
        r = dict(station=st, name=g.name.iloc[0])
        for c, f, s in cols:
            r[f'{c}_{f}_{short[s]}'] = g[(g.clock == c) & (g.frames == f) & (g.source == s)].nse.iloc[0]
        out.append(r)
    o = pd.DataFrame(out)
    m = o.mean(numeric_only=True)
    m['station'], m['name'] = 'mean', ''
    o = pd.concat([o, m.to_frame().T])
    num = o.columns[2:]
    o[num] = o[num].astype(float).round(3)
    o.to_csv(O / 'e3/e3_nse_table.csv', index=False)


def e4_summary():
    d = pd.read_csv(O / 'e4/e4_fixed_grid_long.csv')
    rows = []
    for (m, c), g in d.groupby(['model', 'cells'], sort=False):
        tp, fp, fn = g.tp.sum(), g.fp.sum(), g.fn.sum()
        a = g[g.event == 120].iloc[0]
        rows.append(dict(model=m, cells=c, april2013_rel_l2_pct=100 * a.mean_relative_l2, april2013_rmse_m=a.mean_rmse_m,
                         four_event_rel_l2_pct=100 * g.mean_relative_l2.mean(), four_event_rmse_m=g.mean_rmse_m.mean(),
                         csi=tp / (tp + fp + fn), precision=tp / (tp + fp), recall=tp / (tp + fn), bias=(tp + fp) / (tp + fn),
                         fp_outside_event_mask=g.fp_outside_event_mask.sum() if c == 'fixed_grid' else None,
                         fp_outside_share=(g.fp_outside_event_mask.sum() / fp) if c == 'fixed_grid' else None))
    pd.DataFrame(rows).round(4).to_csv(O / 'e4/e4_summary.csv', index=False)


def d1_tables():
    d = pd.concat([pd.read_csv(O / 'e4/e4_fixed_grid_long.csv'), pd.read_csv(O / 'd1/d1_fixed_grid_long.csv')])
    order = ['C-STRIDE vanilla', 'C-STRIDE terrain', 'C-STRIDE forcing', 'C-STRIDE full', 'CLDNet', 'CLDNet + LD-EnSF']
    rows, fa = [], []
    for m in order:
        for c in ['fixed_grid', 'event_mask']:
            g = d[(d.model == m) & (d.cells == c)].set_index('event')
            a = g.loc[120]
            r = dict(model=m, cells=c)
            for e in [107, 108, 109, 120]:
                r[f'rel_l2_{e}_pct'] = 100 * g.loc[e, 'mean_relative_l2']
            r['four_event_rel_l2_pct'] = 100 * g.mean_relative_l2.mean()
            for e in [107, 108, 109, 120]:
                r[f'rmse_{e}_m'] = g.loc[e, 'mean_rmse_m']
            r['four_event_rmse_m'] = g.mean_rmse_m.mean()
            r.update(april2013_csi=a.csi, april2013_precision=a.precision, april2013_recall=a.recall, april2013_bias=a.bias)
            rows.append(r)
            if c == 'fixed_grid':
                f = dict(model=m, **{f'fp_outside_{e}': int(g.loc[e, 'fp_outside_event_mask']) for e in [107, 108, 109, 120]})
                f['fp_outside_four_events'] = int(g.fp_outside_event_mask.sum())
                fa.append(f)
    pd.DataFrame(rows).round(4).to_csv(O / 'd1/d1_tables5_8_summary.csv', index=False)
    pd.DataFrame(fa).to_csv(O / 'd1/fixed_grid_false_alarms.csv', index=False)


def e6_tables():
    df = pd.read_csv(O / 'e6/e6_scores_long.csv', dtype={'site_no': str})
    srcs = ['SynxFlow'] + [f'C-STRIDE {m} ({v} input)' for m in ['full', 'terrain', 'vanilla', 'four_gauges'] for v in ['sim', 'USGS-CST']]
    out = []
    for kind in ['xs', 'nearest']:
        for subset, mask in [('all', np.ones(len(df), bool)), ('excl_canal_joliet', ~df.site_no.isin(REGULATED))]:
            d = df[(df.cells == kind) & df.valid & mask]
            for s in srcs:
                x = d[d.source == s]
                out.append(dict(cells=kind, subset=subset, source=s, n=len(x), mean_nse=x.nse.mean(), median_nse=x.nse.median(),
                                mean_r=x.r.mean(), mean_alpha=x.alpha.mean(), mean_rmse_m=x.rmse_m.mean(),
                                mean_abs_half_rise_h=x.half_rise_time_h.abs().mean(),
                                mean_abs_centroid_h=x.centroid_time_h.abs().mean()))
            t = d.pivot_table(index='site_no', columns='source', values='nse')
            for m in ['full', 'terrain', 'vanilla', 'four_gauges']:
                a, b = t[f'C-STRIDE {m} (sim input)'], t[f'C-STRIDE {m} (USGS-CST input)']
                out.append(dict(cells=kind, subset=subset, source=f'COUNT usgs>sim {m}', n=len(t), mean_nse=float((b > a).sum()),
                                median_nse=float(np.median(b - a))))
    pd.DataFrame(out).round(3).to_csv(O / 'e6/e6_summary.csv', index=False)

    g = {x['site_no']: x for x in json.loads((O / 'e6/e6_gauges.json').read_text()) if 'excluded' not in x}
    cols = {'SynxFlow': 'SynxFlow', 'C-STRIDE full (sim input)': 'full_sim', 'C-STRIDE full (USGS-CST input)': 'full_usgs',
            'C-STRIDE four_gauges (sim input)': '4g_sim', 'C-STRIDE four_gauges (USGS-CST input)': '4g_usgs'}
    parts = []
    for kind in ['xs', 'nearest']:
        t = df[df.cells == kind].pivot_table(index='site_no', columns='source', values='nse', sort=False)[list(cols)].rename(columns=cols)
        t.columns = [f'{kind}_{c}' for c in t.columns]
        parts.append(t)
    w = pd.concat(parts, axis=1)
    w.insert(0, 'name', [g[s]['station_nm'].replace(', IL', '') for s in w.index])
    w['flags'] = [';'.join(f for f, c in [('regulated', s in REGULATED), ('xs_invalid(2.6km to NHD)', s == '05530000'),
                                          ('nearest_dry<12', g[s]['nearest_dry_before_frame12']),
                                          ('xs_dry<12', g[s]['xs_dry_before_frame12'])] if c) for s in w.index]
    w.loc[w.index == '05530000', [c for c in w.columns if c.startswith('xs_')]] = float('nan')
    w.round(3).to_csv(O / 'e6/e6_nse_table.csv')


def retraining_summary():
    released = summarize(O / 'e5/scores_released_all').assign(run='released')
    e5 = summarize(O / 'e5/scores_retrained').assign(run='E5 corrected Manning')
    e8 = summarize(O / 'e8/scores_e8').assign(run='E8 fixed grid, clean split')
    s = pd.concat([released, e5, e8], ignore_index=True)
    s.insert(0, 'run', s.pop('run'))
    s['model'] = s.model.str.split().str[-1]
    s.to_csv(O / 'retraining_summary.csv', index=False, float_format='%.4f')


if __name__ == '__main__':
    t1_tables()
    persistence()
    e2b_tables()
    e3_table()
    e4_summary()
    d1_tables()
    e6_tables()
    retraining_summary()
