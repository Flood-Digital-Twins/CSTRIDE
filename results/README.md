# Paper results

Numeric data used in the paper are listed below. Newly generated evaluations write to `outputs/`.

| Paper item | Numeric data | Script |
|---|---|---|
| Table 5; Figure 2 | `reconstruction_summary.csv`, `baseline_metrics.csv` `generate_paper_results.py`, `plot_paper_results.py`; baseline values are reported |
| Table 6; Figure 3 | `gauge_simulator_metrics.csv`, `diagnostic_locations.json` `generate_paper_results.py`, `plot_paper_results.py` |
| Table 7; Figures 5–7 | `usgs_metrics.csv` `generate_paper_results.py`, `plot_paper_results.py` |
| Table 8; Figure 4 | `reconstruction_summary.csv`, `baseline_metrics.csv`, `spatial_summary.json` `plot_paper_results.py`, `score_cldnet_baselines.py`; baseline values are reported |
| Table 9; Figures 8–9 | `forecasts.json`, `persistence.json`, `precipitation_mismatch.json` `forecast_observations.py`, `compute_persistence.py`, rainfall plotting scripts |
| Gauge forecast discussion | `gauge_forecasts.json` `forecast_observations.py` |
| Figure 10 | `history_common_support.csv` `generate_paper_results.py`, `plot_paper_results.py` |
| Table 10 | `table10.csv` `evaluate_next_step.py`, `rescore_checkpoints.py` |
| Section 4.6 | `timing.json` `time_inference.py`; bundled values are reported |

Baseline values in `baseline_metrics.csv` are the values reported in Tables 5 and 8, not a claim of newly recomputed scores. Timing values are reported Section 4.6 measurements. LD-EnSF encoder weights are unavailable publicly. The remaining event/lead-level numeric summaries support the corresponding paper plots and tables.
