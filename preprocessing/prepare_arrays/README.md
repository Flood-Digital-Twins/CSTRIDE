# Prepare simulation arrays

`prepare_arrays.py` reads `flow_variables.npy`, `DEM.npy`, `rain_source.npy` and `input/field/manning.dat` from a collected SynxFlow event directory. Defaults: one burn-in frame, 96 retained hours and an event-maximum depth threshold of 0.1 m.

```bash
python preprocessing/prepare_arrays/prepare_arrays.py --input-dir data/sims_30/event_107 --traj-id 107 --output-dir data/Chicago --legacy-manning-order
```

Use `--legacy-manning-order` for released weights; omit it for corrected new training. `--static-precision historical` preserves numbered-event quantization and event-120 raw float32 elevation/slope. `--indices-only` writes raster indices without full fields; `--reference-root` compares reconstructed arrays against a caller-supplied prepared directory. Outputs stay in the specified directory. See [array formats](../../data/README.md).
