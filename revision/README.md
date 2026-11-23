# Revision checks (Journal of Hydrology revision, November 2026)

This folder holds the scripts, result tables and retrained models behind the results added or changed in the revision. Outside `revision/` only the top-level `README.md` and `results/README.md` gained pointers here: the released models, configurations, data and result files are as submitted, so the submission numbers still reproduce with the released scripts.

```
revision/
  scripts/   one script per check, plus paths.py (input and output locations) and make_tables.py (derived tables)
  results/   tables, cross-section cells, observation series and figures, one folder per check
  models/    retrained models E5 (corrected Manning) and E8 (fixed grid, clean split), with SHA256SUMS
```

## Revised paper items

| Paper item | Results (under `results/`) | Scripts (under `scripts/`) |
|---|---|---|
| Table 7: USGS comparison at the CLDNet cross-sections, CST clock | `e3/e3_nse_table.csv` (columns `CST_12-95_*`), `e3/e3_crosssection_scores_long.csv`, cells `e3/e3_crosssection_cells_nhd.json` | `e3_crosssection_scores.py`; cells: `e3_reproduce_table3.py`, `e3_recover_cells.py`, `e3_nhd_crosssections.py` |
| Table 7 timing (half-rise and centroid), r and alpha | `d2/d2_timing_table.csv`, `d2/d2_means.csv`, `d2/d2_crosssection_timing_long.csv` | `d2_crosssection_timing.py` |
| Figures 5-7 on the CST clock | `figures/B/` (cross-section cells, own-mean centring); `figures/A/` (released sensor cells, Table A.3 convention) | `f1_figures.py` |
| Appendix Table A.3: USGS comparison at the released sensor cells | `e1e2/table7_CST_offset12-95.csv` (the submission pairing: `table7_published_offset12-95.csv`; CDT sensitivity: `table7_CDT_offset12-95.csv`), `e1e2/summary_means.csv`, `e1e2/e1e2_usgs_metrics_long.csv` | `e1_usgs_clock.py`, `e1_tables.py` |
| Timing at the sensor cells, 1 h persistence, split-sample offset | `t1/t1_timing_table.csv`, `t1/t1_r_alpha_table.csv`, `t1/persistence_1h.csv`; `e2b/e2b_nse_table.csv`, `e2b/e2b_rmse_table.csv`; causal offsets: `e1e2/e2_*_inputcausal_*.csv` | `e1b_timing_split.py`, `make_tables.py` |
| USGS input series, CST clock | `e1e2/usgs_observations_CST_full.npz` (submission pairing: `usgs_observations_published_full.npz`, identical to `data/gauges/usgs_2013/usgs_observations.npz`) | `e1_usgs_clock.py` |
| Tables 5 and 8, Appendix Tables A.1-A.2, Section 4.1: fixed CLDNet grid | `d1/d1_tables5_8_summary.csv` (six models, per event and four-event, fixed grid and event masks, April 2013 CSI, precision, recall, bias), `d1/fixed_grid_false_alarms.csv`, `e4/e4_summary.csv`; long tables `e4/e4_fixed_grid_long.csv`, `d1/d1_fixed_grid_long.csv` | `e4_fixed_grid.py`, `d1_fixed_grid_more.py`, `make_tables.py` |
| Appendix Tables A.4-A.5: independent USGS gauges | `e6/e6_nse_table.csv` (per gauge, with flags), `e6/e6_summary.csv`, gauge list with exclusions `e6/e6_gauges.json`; robustness and amplitude diagnostics `e7/e7ab_summary.csv`, `e7/e7c_dnse_vs_inputs.csv` | `e6_independent_gauges.py`, `e7_diagnostics.py`, `make_tables.py` |
| Appendix B, Table B.1: retraining | `retraining_summary.csv` (released vs E5 vs E8, all metrics); `e5/scores_retrained/`, `e8/scores_e8/`, released models rescored with the same code: `e5/scores_released_all/` | `e5_prepare_data.py`, `e8_prepare_data.py`, `e8_train_resumable.py`, `e5_score.py`, `e8_score.py`, `summarize_scores.py` |
| Manning channel check | `manning_verification.md` | |

Frames 12-95 of the April 2013 event (storm 120) are scored throughout. Four-event values are means over storms 107, 108, 109 and 120 of the per-event mean snapshot values. CSI, precision, recall and bias use a 0.5 m depth threshold, pooled over frames.

## Conventions and findings documented here

**USGS clock.** The raw stage files in `data/gauges/usgs_2013/raw_stage/` have naive timestamps in CST (UTC-6). This was checked sample by sample against api.waterdata.usgs.gov for all six input gauges. Simulation frame *i* is 2013-04-17 08:00 UTC + *i* h, so it pairs with CSV time 02:00 + *i*. The submission paired frame *i* with CSV 08:00 + *i*, which puts the simulation about 6 h late; the CLDNet study (Si et al., Table 3) paired it in CDT, 5 h late (reproduced in `e3/e3_synxflow_crosssection_vs_usgs.csv`). The released default `data/gauges/usgs_2013/usgs_observations.npz` keeps the submission pairing. To run `scripts/predict_cstride.py --observations usgs` on the CST clock, copy `revision/results/e1e2/usgs_observations_CST_full.npz` over that file in your working copy (only `input_depth` is read).

**Cross-sections.** Table 7 now compares water-surface elevation at the CLDNet notebook's cross-sections: a 60 or 90 m line perpendicular to the nearest NHD medium-resolution flowline, keeping the first 2-3 rasterised cells, with WSE = maximum depth over the cells + mean DEM. The cells were rebuilt from NHD and checked against an exact match of the stored SynxFlow series (`e3/e3_crosssection_cells_nhd.json`). Each series is centred on its own 12-95 mean. The last row of `e3/e3_nse_table.csv` is the mean over the six gauges, so count gauges from the other rows or from the long table.

**Fixed grid.** Tables 5 and 8 are scored on CLDNet's fixed 1,408,587-cell Des Plaines grid (`data/illinois_grid/grid.npz`, `aggregate_mask`, in the CLDNet repository) instead of each event's wet-cell mask, so all models are compared on the same cells. Event-mask values remain in the tables (`cells = event_mask`).

**Manning channel.** The released prepared arrays use the legacy, row-mirrored Manning channel (`--legacy-manning-order` in `preprocessing/prepare_arrays/`), and the released models were trained and are evaluated with it. The corrected channel equals the processed land cover in SynxFlow's bottom-up cell numbering (`manning_verification.md`). E5 and E8 use the corrected channel.

**Released split.** In `splits/chicago_split.json`, `checkpoint_monitor_files` (used for `best_model.ckpt` and early stopping) are storms 107-116. They include three of the four test storms (107, 108 and 109); 110-116 are not trained on. `train_files` has 100 entries: the 90 training storms plus 85-89 and 95-99, whose rainfall is identical to storms 25-29 and 35-39, so ten storms are seen twice per epoch. E8 trains on the 90 distinct storms and uses no test storm for selection.

## Retrained models

| Run | What changed from the released protocol | Checkpoint | Epochs completed | Checkpoint epoch |
|---|---|---|---|---|
| E5 full | Manning channel corrected; same configuration, seed, normalizers, monitor storms and early-stopping rule | `models/e5/full/best_model.ckpt` | 180 (early stop) | 150 |
| E5 terrain | as E5 full | `models/e5/terrain/best_model.ckpt` | 200 (early stop) | 200 |
| E8 full | CLDNet fixed grid, corrected Manning, 90 distinct training storms, normalizers refitted on them, one cell per gauge, fixed epochs, no selection | `models/e8/full/final_model.ckpt` | 180 | 180 |
| E8 terrain | as E8 full | `models/e8/terrain/final_model.ckpt` | 210 | 210 |
| E8 vanilla | as E8 full | `models/e8/vanilla/final_model.ckpt` | 180 | 180 |
| E8 forcing | as E8 full | `models/e8/forcing/final_model.ckpt` | 150 | 150 |

The E8 epoch counts equal the epochs the released runs completed. Each model folder has `config.yaml`, `sensor_indices.json`, `normalizers.npz` (E5: the released normalizers; E8: refitted) and `training_summary.json`. Each configuration is a single run with seed 0.

Four-event relative L2 on the fixed grid, in % (`retraining_summary.csv`; CLDNet 16.94):

| Model | Released | E5 | E8 |
|---|---|---|---|
| full | 12.94 | 12.43 | 12.02 |
| terrain | 16.48 | 16.33 | 12.65 |
| vanilla | 22.78 | | 19.16 |
| forcing | 20.60 | | 17.69 |

## Rerunning

Scripts read their inputs through `scripts/paths.py`, which takes each location from an environment variable (defaults follow this repository's layout). Run them from the repository root, for example `python revision/scripts/make_tables.py`. Outputs go to `revision/results/` (or `REVISION_OUTPUTS`), so `git diff` shows any change.

Inputs that are not distributed here:

- `SYNXFLOW_SIMS`: the dense SynxFlow simulations (one folder per storm with `flow_variables.npy`), needed for every field score and for the E5/E8 data.
- `CLDNET_ROOT`: a clone of github.com/Flood-Digital-Twins/CLDNet (fixed grid and forcings). For E4, first write the released CLDNet fields from inside that clone: `python scripts/predict_cldnet.py --storm <s> --save-fields --out-dir <this repo>/revision/results/e4/cldnet` for `<s>` = 107, 108, 109 and 2013-04-17_2013-04-21 (about 770 MB each, not committed).
- `USGS_2013_GAUGES`: the CLDNet study's April 2013 USGS folder. Its gauge list and stage files are in the CLDNet repository; `dem_5070.tif`, `truth_traj120.npy` and `output/` come from the study's validation notebook.
- `LDENSF_PREDICTIONS`: saved CLDNet + LD-EnSF fields (D1 only; the LD-EnSF encoder weights are not public).
- `CSTRIDE_ARRAYS` and `CSTRIDE_PAPER_RESULTS`: the prepared arrays and paper products of the main workflow (`preprocessing/`, `scripts/generate_paper_results.py`).

Order: `e1_usgs_clock.py` and `e1_tables.py` (run inside `results/e1e2`); `e1b_timing_split.py`; `e3_reproduce_table3.py`, `e3_recover_cells.py`, `e3_nhd_crosssections.py`, `e3_crosssection_scores.py`; `d2_crosssection_timing.py`; `e4_fixed_grid.py`, `d1_fixed_grid_more.py`; `e6_independent_gauges.py`, `e7_diagnostics.py`; `f1_figures.py`; then `make_tables.py`. `make_tables.py` alone rebuilds every summary table from the long tables shipped here.

Retraining, one GPU per run (an E8 epoch took about 258 s on an H100):

```
python revision/scripts/e5_prepare_data.py          # corrected-Manning arrays -> results/e5/data/Chicago
python revision/scripts/e8_prepare_data.py          # fixed-grid arrays, split and gauge cells -> results/e8/
E8_STATE=outputs/e8_full_state.pt python revision/scripts/e8_train_resumable.py \
    --config revision/results/e8/configs/full_e8.yaml --device cuda:0
python revision/scripts/e8_score.py --ckpt full=revision/models/e8/full/final_model.ckpt   # repeat --ckpt per model
python revision/scripts/e5_score.py --full revision/models/e5/full/best_model.ckpt \
    --terrain revision/models/e5/terrain/best_model.ckpt
python revision/scripts/summarize_scores.py revision/results/e8/scores_e8
```

`e8_train_resumable.py` runs the released trainer (`cstride.train_next_step`) with a loop that saves model, optimizer, scheduler and RNG states to `E8_STATE` after every epoch and resumes from it. With `E8_WALL_SECONDS` set it exits with code 3 before that budget runs out, so a dependent job can continue. With a configured `patience` (the E5 configurations) it applies the released early-stopping rule; with `patience: null` (E8) it trains the fixed number of epochs. After a resume the data-loader workers restart their random streams from their seeds; this is logged. The released models were rescored with `e5_score.py --legacy --tag released_all --full checkpoints/full/best_model.ckpt --terrain checkpoints/terrain/best_model.ckpt --extra vanilla=checkpoints/vanilla/best_model.ckpt --extra forcing=checkpoints/forcing/best_model.ckpt`, which reproduces E4 and D1 exactly.
