# Trained models

The paper uses `best_model.ckpt`; final weights are also retained. Available gauge-prediction LSTMs accompany each model. Configurations and safe sensor maps are local to each model folder.

| Model | Best epoch | Completed epochs | Configured maximum |
|---|---:|---:|---:|
| full | 180 | 180 | 700 |
| vanilla | 180 | 180 | 700 |
| forcing | 150 | 150 | 700 |
| terrain | 180 | 210 | 700 |
| lag1 | 120 | 170 | 700 |
| lag4 | 220 | 220 | 700 |
| lag6 | 170 | 200 | 700 |
| lag24 | 180 | 180 | 700 |
| four_gauges | 210 | 220 | 700 |
| dropout050 | 170 | 190 | 700 |
| noise005 | 170 | 180 | 700 |
| noise010 | 190 | 190 | 700 |
| noise020 | 120 | 170 | 700 |

The configured maximum is not the number of completed epochs. Model weights and their internal forecaster architecture metadata are preserved. `manifest.json` contains current integrity hashes. Portable training configurations are in `../configs/next_step/`.

The latent/direct autoregressive modules are extra workflows; the paper forecast uses the gauge forecaster. Copy a model into `outputs/` before running evaluators that append files beside a checkpoint.
