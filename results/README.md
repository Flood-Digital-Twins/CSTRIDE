# Paper results

Numeric data used in the paper are listed below. The Script column links to the corresponding evaluation and plotting tools; newly generated evaluations write to `outputs/`.

| Paper item | Numeric data | Script |
|---|---|---|
| Table 5; Figure 2 | [reconstruction_summary.csv](reconstruction_summary.csv), [baseline_metrics.csv](baseline_metrics.csv) | [generate_paper_results.py](../scripts/generate_paper_results.py), [plot_paper_results.py](../scripts/plot_paper_results.py) |
| Table 6; Figure 3 | [gauge_simulator_metrics.csv](gauge_simulator_metrics.csv), [diagnostic_locations.json](diagnostic_locations.json) | [generate_paper_results.py](../scripts/generate_paper_results.py), [plot_paper_results.py](../scripts/plot_paper_results.py) |
| Table 7; Figures 5–7 | [usgs_metrics.csv](usgs_metrics.csv) | [generate_paper_results.py](../scripts/generate_paper_results.py), [plot_paper_results.py](../scripts/plot_paper_results.py) |
| Table 8; Figure 4 | [reconstruction_summary.csv](reconstruction_summary.csv), [baseline_metrics.csv](baseline_metrics.csv), [spatial_summary.json](spatial_summary.json) | [plot_paper_results.py](../scripts/plot_paper_results.py), [score_cldnet_baselines.py](../scripts/score_cldnet_baselines.py) |
| Table 9; Figures 8–9 | [forecasts.json](forecasts.json), [persistence.json](persistence.json), [precipitation_mismatch.json](precipitation_mismatch.json) | [forecast_observations.py](../scripts/forecast_observations.py), [compute_persistence.py](../scripts/compute_persistence.py), [plot_chicago_precipitation_mismatch.py](../scripts/plot_chicago_precipitation_mismatch.py), [plot_chicago_future_precipitation.py](../scripts/plot_chicago_future_precipitation.py) |
| Gauge forecast discussion | [gauge_forecasts.json](gauge_forecasts.json) | [forecast_observations.py](../scripts/forecast_observations.py) |
| Figure 10 | [history_common_support.csv](history_common_support.csv) | [generate_paper_results.py](../scripts/generate_paper_results.py), [plot_paper_results.py](../scripts/plot_paper_results.py) |
| Table 10 | [table10.csv](table10.csv) | [evaluate_next_step.py](../scripts/evaluate_next_step.py), [rescore_checkpoints.py](../release/rescore_checkpoints.py) |
| Section 4.6 | [timing.json](timing.json) | [time_inference.py](../scripts/time_inference.py) |

Baseline values in `baseline_metrics.csv` are the values reported in Tables 5 and 8, not a claim of newly recomputed scores. Timing values are reported Section 4.6 measurements. The linked baseline and timing scripts generate new evaluations; measurements can differ from the bundled paper values. LD-EnSF encoder weights are unavailable publicly. The remaining event/lead-level numeric summaries support the corresponding paper plots and tables.

## Revision (November 2026)

Results added or changed in the revision are under [../revision/results/](../revision/results), with their scripts in [../revision/scripts/](../revision/scripts); the files above are unchanged. Details and rerun instructions: [../revision/README.md](../revision/README.md).

| Revised paper item | Numeric data | Script |
|---|---|---|
| Table 7 (CLDNet cross-sections, CST clock) | [e3_nse_table.csv](../revision/results/e3/e3_nse_table.csv), [e3_crosssection_cells_nhd.json](../revision/results/e3/e3_crosssection_cells_nhd.json), timing [d2_timing_table.csv](../revision/results/d2/d2_timing_table.csv) | [e3_crosssection_scores.py](../revision/scripts/e3_crosssection_scores.py), [d2_crosssection_timing.py](../revision/scripts/d2_crosssection_timing.py) |
| Figures 5-7 (CST clock) | [figures/B](../revision/results/figures/B), sensor-cell variant [figures/A](../revision/results/figures/A) | [f1_figures.py](../revision/scripts/f1_figures.py) |
| Appendix Table A.3 (sensor cells, CST clock) | [table7_CST_offset12-95.csv](../revision/results/e1e2/table7_CST_offset12-95.csv), [t1_timing_table.csv](../revision/results/t1/t1_timing_table.csv), [persistence_1h.csv](../revision/results/t1/persistence_1h.csv), [e2b_nse_table.csv](../revision/results/e2b/e2b_nse_table.csv) | [e1_usgs_clock.py](../revision/scripts/e1_usgs_clock.py), [e1b_timing_split.py](../revision/scripts/e1b_timing_split.py), [make_tables.py](../revision/scripts/make_tables.py) |
| Tables 5 and 8, Appendix Tables A.1-A.2, Section 4.1 (fixed CLDNet grid) | [d1_tables5_8_summary.csv](../revision/results/d1/d1_tables5_8_summary.csv), [fixed_grid_false_alarms.csv](../revision/results/d1/fixed_grid_false_alarms.csv), [e4_summary.csv](../revision/results/e4/e4_summary.csv) | [e4_fixed_grid.py](../revision/scripts/e4_fixed_grid.py), [d1_fixed_grid_more.py](../revision/scripts/d1_fixed_grid_more.py) |
| Appendix Tables A.4-A.5 (independent USGS gauges) | [e6_nse_table.csv](../revision/results/e6/e6_nse_table.csv), [e6_summary.csv](../revision/results/e6/e6_summary.csv), [e6_gauges.json](../revision/results/e6/e6_gauges.json), [e7ab_summary.csv](../revision/results/e7/e7ab_summary.csv) | [e6_independent_gauges.py](../revision/scripts/e6_independent_gauges.py), [e7_diagnostics.py](../revision/scripts/e7_diagnostics.py) |
| Appendix B, Table B.1 (retraining E5 and E8) | [retraining_summary.csv](../revision/results/retraining_summary.csv), models in [../revision/models/](../revision/models) | [e8_train_resumable.py](../revision/scripts/e8_train_resumable.py), [e5_score.py](../revision/scripts/e5_score.py), [e8_score.py](../revision/scripts/e8_score.py) |
