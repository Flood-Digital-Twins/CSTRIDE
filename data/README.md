# Data formats

The dataset contains 94 events: 90 training events and 4 test events. Test IDs are 107, 108, 109 and 120. Full simulated fields are supplied separately; compact inputs support released-model inference.

| Input | Contents |
|---|---|
| `gauges/simulated_depth.npz` | Event IDs, station IDs and 96 hourly depths at six gauges; depth shape `(94,96,6)` |
| `gauges/USGS_6sensors.json`, `USGS_4sensors.json` | Event-ID to gauge-cell index mappings |
| `gauges/usgs_2013/` | Raw stages, mean-aligned WSE and retrospective depth inputs for event 120 |
| `grids/grid_traj<ID>.npz` | Stored coordinates, elevation, Manning and slope for each test event, plus exact raster indices |
| `normalization/normalizers.npz` | Training min/max values for state, depth, rainfall and static features |
| `storms/storm_periods.csv` | Rainfall periods and locations; see [column definitions](storms/README.md) |
| `SHA256SUMS` | Integrity hashes for compact inputs and metadata |

Rainfall comes from the separate [CLDNet](https://github.com/Flood-Digital-Twins/CLDNet) dataset. Read rows 0–95 of `data/forcings/event_<ID>/rain_source.npy`; event 120 uses `event_2013-04-17_2013-04-21`. Preserve float32-to-float16 conversion before normalization. Event 107 matches the prepared rainfall exactly under this conversion; the checked public CLDNet revision is `5178e3811df7b2ceaa5ff854f506e3ef1e8b439f`. Training min/max scaling maps channels to `[-1,1]`; constant channels have scale 1, and coordinates pass through unchanged. Do not refit on test data.

## Prepared arrays

Training/scoring requires a `manifest.json` containing `time_steps` and a `files` list. Each entry has `traj_id` and relative filenames `state_file`, `depth_file`, `coords_file`, `precip_file`, `elevation_file`, `manning_file`, `slope_file`.

| Array | Shape |
|---|---|
| State | `(T,N,C)` or `(1,T,N,C)` |
| Depth fallback | `(T,N,1)` |
| Coordinates | `(N,2)` |
| Rainfall | `(T,507)` |
| Static features | `(N,)` |

Reference `T=96`; `N` varies by event. Depth is target channel 0, and all arrays share point ordering. The [converter](../preprocessing/prepare_arrays/README.md) removes one burn-in frame and retains cells with finite depth throughout and event maximum depth ≥0.1 m.

Released coordinates are float16 and retain quantization. Preserve the as-trained Manning registration when using existing weights. Numbered events retain elevation reconstructed from quantized DEM z-scores and quantized slope; event 120 uses separately converted raw float32 elevation/slope. USGS inputs retain retrospective mean alignment over indices 12–95. Rainfall/depth timing preserves the original burn-in convention.

Released data use [CC BY 4.0](LICENSE.md). See the [main README](../README.md) for commands and input qualifications.
