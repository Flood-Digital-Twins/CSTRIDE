# Storm metadata

`storm_periods.csv` lists all 94 events using original event IDs. Rainfall time stamps are UTC; Stage IV values are accumulations over the hour ending at the stamped time. Rainfall at index `k` drives simulation hour `k` to `k+1`.

| Columns | Meaning |
|---|---|
| `traj_id`, `cldnet_event` | Event ID and CLDNet forcing folder |
| `start_utc`, `end_utc` | First/last time stamps of the 97 hourly rainfall fields |
| `on_basin`, `site` | Whether rainfall occurred on the basin, and its recorded location |
| `field_centre_lat`, `field_centre_lon`, `km_from_basin` | Matched field centre and distance from the basin |
| `mean_depth_mm`, `peak_mm_per_h` | Grid-mean depth over 96 hours and peak hourly intensity |
| `corr_with_stage4`, `corr_at_basin` | Correlation with Stage IV at the matched location and at the basin |
| `hours_compared`, `hours_without_archive_data` | Hours compared and unavailable archive hours |

The 2013 event (120) occurred on the basin; numbered events were recorded elsewhere and used as forcing. Locations are matched against Stage IV using the six wettest hours and are accurate to a few kilometres.
