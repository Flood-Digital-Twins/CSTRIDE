"""Summarize a scores_* folder (e4_long.csv, table7_crosssection_long.csv, e6_long.csv) in the reporting format.

Aggregation matches results/e4/e4_summary.csv, results/d1/d1_tables5_8_summary.csv and results/e6/e6_summary.csv:
  four-event relative L2 / RMSE = mean of the four per-event means (frames 12-95); CSI, precision, recall, bias
  at 0.5 m for April 2013 (event 120) pooled over frames; false alarms outside the event mask summed over events.
  T7: mean NSE over the six input gauges (Si et al. cross-sections, CST, 12-95), count of gauges where USGS > sim.
  E6: the 19 valid cross-section gauges (valid == True, cells == xs), mean NSE and count USGS > sim.
Usage: summarize_scores.py <scores dir> [<scores dir> ...]  -> writes <dir>/summary.csv and prints it.
"""
import sys
from pathlib import Path

import pandas as pd


def summarize(d):
    d = Path(d)
    e4 = pd.read_csv(d / 'e4_long.csv')
    t7 = pd.read_csv(d / 'table7_crosssection_long.csv')
    e6 = pd.read_csv(d / 'e6_long.csv')
    rows = []
    for m in e4.model.unique():
        key = m.split()[-1]  # 'E8 full' / 'C-STRIDE full' / 'full' -> 'full'
        r = dict(model=m)
        fg = e4[(e4.model == m) & (e4.cells == 'fixed_grid')].set_index('event')
        em = e4[(e4.model == m) & (e4.cells == 'event_mask')].set_index('event')
        for ev in [107, 108, 109, 120]:
            r[f'fixed_rel_l2_{ev}_pct'] = 100 * fg.loc[ev, 'mean_relative_l2']
        r['fixed_four_event_rel_l2_pct'] = 100 * fg.mean_relative_l2.mean()
        r['fixed_four_event_rmse_m'] = fg.mean_rmse_m.mean()
        r['fixed_april_rel_l2_pct'] = 100 * fg.loc[120, 'mean_relative_l2']
        r['fixed_april_rmse_m'] = fg.loc[120, 'mean_rmse_m']
        for k in ('csi', 'precision', 'recall', 'bias'):
            r[f'april_{k}'] = fg.loc[120, k]
        r['event_mask_four_event_rel_l2_pct'] = 100 * em.mean_relative_l2.mean()
        r['event_mask_four_event_rmse_m'] = em.mean_rmse_m.mean()
        r['fp_outside_event_mask_four_events'] = fg.fp_outside_event_mask.sum()
        r['fp_outside_event_mask_april'] = fg.loc[120, 'fp_outside_event_mask']
        tm = t7[t7.model.str.split().str[-1] == key]
        if len(tm):
            p = tm.pivot(index='station', columns='input', values='nse')
            r['t7_mean_nse_sim'] = p['sim'].mean()
            r['t7_mean_nse_usgs'] = p['USGS-CST'].mean()
            r['t7_usgs_improves'] = int((p['USGS-CST'] > p['sim']).sum())
            r['t7_n'] = len(p)
        g = e6[(e6.model.str.split().str[-1] == key) & (e6.cells == 'xs') & e6.valid]
        if len(g):
            p = g.pivot(index='site_no', columns='input', values='nse')
            r['e6_mean_nse_sim'] = p['sim'].mean()
            r['e6_mean_nse_usgs'] = p['USGS-CST'].mean()
            r['e6_usgs_wins'] = int((p['USGS-CST'] > p['sim']).sum())
            r['e6_n'] = len(p)
        rows.append(r)
    s = pd.DataFrame(rows)
    for c in [c for c in s.columns if c.startswith('fp_') or c.endswith(('_improves', '_wins', '_n'))]:
        s[c] = s[c].astype('Int64')  # counts
    s.to_csv(d / 'summary.csv', index=False, float_format='%.4f')
    return s


if __name__ == '__main__':
    pd.set_option('display.width', 250, 'display.max_columns', 40)
    for d in sys.argv[1:]:
        print('==', d)
        print(summarize(d).set_index('model').T.round(4).to_string())
