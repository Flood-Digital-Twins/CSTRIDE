# Manning channel check (2026-11-21)

Question: is the "corrected" Manning registration (prepare_arrays.py without --legacy-manning-order), used for E5 and E8, the one the simulator used?

1. **Simulator setting.** `run_hipims_event.py` with `variable_manning=0` (as in every `simulation_config.json`): Manning 0.02 where `data_for_HiPIMS_30/landcover_5070_processed.tif` is 1 (NLCD classes 11/12, open water and ice), 0.05 elsewhere.
2. **Cell numbering in SynxFlow.** `synxflow/IO/indep_functions.py::_get_cell_id_array` numbers valid cells "from bottom, left to right, top" (cumulative sum over `np.flipud(dem)`). `manning.dat` lists ids 0..4,188,839 in order. The corrected reader fills values in row-major order of the flipped raster, which is this numbering; the legacy reader fills from the top row.
3. **Against the land cover.** `DEM.npy` equals `dem_5070.tif` (north-up, same transform). On all 4,188,840 valid cells, the corrected Manning raster equals the processed land cover exactly (0 mismatches). The legacy raster mismatches at 150,238 cells (3.6 %). Both have a 1.82 % water share; the legacy one is mirrored in rows.
4. **Against the hydraulics.** Event 107: cells marked water by the corrected channel are wet (depth >= 0.1 m) at 56.5 % at frame 1 and 74.4 % at frame 48 (median depth 0.22 m). Cells marked water by the legacy channel are wet at 9.4 % and 14.8 % (median 0.002 m), no more than land cells (14.0-15.2 %).

Conclusion: the corrected channel is right; the legacy channel used by every released C-STRIDE and CLDNet checkpoint is the mis-registered one. On the fixed 1,408,587-cell grid, the corrected water share is 5.02 %, matching the CLDNet grid README.
