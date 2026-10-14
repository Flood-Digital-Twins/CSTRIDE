# C-STRIDE: basin-wide flood prediction from sparse gauge histories

Code, trained models and compact inputs for Yanjie Tong, Phillip Si, Yuan Qiu, and Peng Chen (2026), *C-STRIDE: An Observation-Driven AI Digital Twin for Predicting Basin-Wide Flood Fields from Sparse Stream-Gauge Histories*. [Paper](https://arxiv.org/abs/2609.39005). Repository: [Flood-Digital-Twins/CSTRIDE](https://github.com/Flood-Digital-Twins/CSTRIDE).

C-STRIDE uses an LSTM to encode gauge histories and a modulated neural field to predict depth. Rainfall and terrain are optional conditioning inputs. Multi-step forecasting combines the fixed next-step model with a gauge-prediction LSTM. See [Figure 1](figures/fig01_architecture.pdf).

The dataset contains **94 events: 90 training events and 4 test events**. Original test event IDs are 107, 108, 109 and 120. Compact gauge histories, query grids and cached normalizers allow inference without the full simulations. Rainfall is read from the public [CLDNet](https://github.com/Flood-Digital-Twins/CLDNet) dataset.

## Installation and prediction

Use Python 3.10 or later with a suitable PyTorch CPU/CUDA build. Commands run from the repository root; set explicit paths for external data.

```bash
python -m pip install -r requirements.txt
python scripts/predict_cstride.py --model full --storm 120 --cldnet-root ../CLDNet --device cuda:0
python scripts/predict_cstride.py --model full --storm 120 --cldnet-root ../CLDNet --observations usgs --device cuda:0
python scripts/predict_cstride.py --model full --storm 120 --cldnet-root ../CLDNet --forecast --obs-rainfall --start 12 --steps 24 --device cuda:0
# CPU inference without rainfall:
python scripts/predict_cstride.py --model vanilla --storm 120 --steps 1 --device cpu
```

The next-step prediction for time `t` uses observations from `t-lag` through `t-1`. Forecasting inserts predicted gauges and supplied future rainfall into subsequent histories. USGS inputs are available for event 120. Vanilla and terrain-only next-step models do not require rainfall.

Outputs go under `outputs/`; `--save-fields` adds per-step fields and `--chunk-size` bounds decoding memory. The compact launcher uses the supplied histories without adding robustness noise/dropout. Tracking is disabled by default and requires no account. Python imports use `cstride`; the model classes are `CSTRIDE` and `CSTRIDEForecast`.

## Data and preprocessing

[Data formats](data/README.md) describe the compact gauge histories, safe sensor maps, query grids, USGS alignment and cached normalization. Full simulations are supplied separately when training or scoring. Prepare collected SynxFlow output as follows:

```bash
python preprocessing/prepare_arrays/prepare_arrays.py --input-dir data/sims_30/event_107 --traj-id 107 --output-dir data/Chicago --legacy-manning-order
```

Repeat for events needed by the split. The legacy Manning switch reproduces the feature used by released weights; omit it for corrected new training. See [preprocessing options](preprocessing/prepare_arrays/README.md). Original event IDs are retained in [the split](splits/chicago_split.json); [storm metadata](data/storms/README.md) explains rainfall periods and locations.

## Training and evaluation

The thirteen model configurations are in `configs/next_step/`; [model details](checkpoints/README.md) list checkpoints and epochs. Training uses SOAP, normalized MSE, a plateau scheduler and checkpoint-loss monitoring with early stopping.

```bash
python scripts/train_next_step.py --config configs/next_step/full.yaml --data-root data/Chicago --device cuda:0
mkdir -p outputs
cp -r checkpoints/full outputs/full
python scripts/evaluate_next_step.py --path outputs/full --data-root data/Chicago --normalizers data/normalization/normalizers.npz --device cuda:0
python scripts/forecast_observations.py --path outputs/full --data-root data/Chicago --normalizers data/normalization/normalizers.npz --horizon 24 --obs_include_precipitation_input --device cuda:0
```

Use a checkpoint copy for evaluators that append files beside it. Training normally fits normalization from the training data; `--normalizers` loads cached statistics. `--split-file` selects an explicit event-ID protocol; `--num-workers 0 --device cpu` is useful for debugging. Compilation is opt-in. Run a launcher with `--help` for its complete options.

For multi-GPU training:

```bash
torchrun --nproc_per_node=2 scripts/train_next_step.py --config configs/next_step/full.yaml --data-root data/Chicago --ddp
```

The GPU filter defaults to 40 GiB free memory. `--init-checkpoint` loads weights without optimizer/scheduler state and writes to that checkpoint's run directory, so use a working copy. Direct autoregressive and latent forecasts are extra model workflows; the paper forecast uses the gauge forecaster.

## Paper results and figures

[Results](results/README.md) maps paper tables to numeric files; [figures](figures/README.md) lists Figures 1–10. New evaluations write under `outputs/` and remain separate from reported paper values.

```bash
python release/rescore_checkpoints.py --data-root data/Chicago --device cuda:0
python scripts/generate_paper_results.py --data-root data/Chicago --device cuda:0
python scripts/plot_paper_results.py
python scripts/compute_persistence.py --data-root data/Chicago
python scripts/plot_chicago_future_precipitation.py --persistence results/persistence.json
python scripts/plot_chicago_precipitation_mismatch.py
python scripts/time_inference.py --cldnet-root ../CLDNet --device cuda:0
```

The generator writes prediction products to `outputs/paper/`, which the plotting script reads. The rainfall plots can use the bundled paper summaries without full simulations. Supply `--data-root` to recompute rainfall mismatch statistics. Re-scoring applies each model's configured noise/dropout and records an explicit seed. Noise scores may vary with the seed. Add `--forecast-data-root data/Chicago` to the timing command to measure the complete 40-by-24-step forecast experiment; its timing includes preparation/scoring, whereas next-step timing excludes scoring.

For public CLDNet predictions, run its `predict_cldnet.py --save-fields`, then adapt each output and score it against matching prepared truth:

```bash
python scripts/adapt_cldnet_predictions.py --input ../CLDNet/outputs/predictions/cldnet_107.npz --grid ../CLDNet/data/illinois_grid/grid.npz --output outputs/cldnet_predictions/predictions_traj107.npz
python scripts/score_cldnet_baselines.py --prediction_dir outputs/cldnet_predictions --data_root data/Chicago --source_dense_roots data/sims_30 --valid_mask_mode dense_flow --trajectory_ids 107 108 109 120 --output outputs/cldnet_scores.txt
```

Use the actual CLDNet output filename and repeat conversion for all four test events. Zero-filled cells outside CLDNet's aggregate mask are not model predictions; the event-120 grid differs by two cells, so converted predictions need not reproduce the reported baseline exactly. Bundled baseline values are transcribed from Tables 5 and 8; the CLDNet comparison uses its epoch-539 checkpoint.

## Verification

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python release/verify_release.py --checksums
sha256sum -c data/SHA256SUMS
```

Tests cover released weights, gauge forecasters, temporal alignment, normalization, decoder chunking, event-ID loading, publication assets and synthetic training/evaluation. Full training convergence, multi-GPU training and compilation are outside this verification. Checkpoint and compact-input hashes are recorded in their manifests.

## Input conventions and licensing

Historical float16 coordinates and the as-trained Manning feature are preserved for checkpoint compatibility. Corrected Manning registration should be used for new training. Numbered rainfall events were recorded elsewhere and applied to the basin; the 2013 event occurred on the basin. USGS inputs retain retrospective mean alignment. Public LD-EnSF encoder weights are unavailable, so its reported baseline cannot currently be regenerated solely from public assets.

Original code and model weights use [MIT](LICENSE); released data use [CC BY 4.0](data/LICENSE.md). Third-party notices are retained in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
