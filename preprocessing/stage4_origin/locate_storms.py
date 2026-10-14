#!/usr/bin/env python3
"""When and where did the rain of each Des Plaines storm actually fall?

Every storm of the Des Plaines dataset is a 97-hour NCEP Stage IV extract with real time stamps
(CLDNet repository, data/forcings/event_<id>/event_<id>.nc). The x/y coordinates in those files all
describe the Des Plaines basin, although most fields were recorded elsewhere in the Midwest. This
script compares each storm with the Stage IV archive at the stamped hours:

  1. at the basin, with the coordinates written in the file;
  2. everywhere in the central and eastern US, to find the place where the field was recorded.

It writes one row per storm: the period (UTC), the centre of the place where the field matches the
archive, its distance from the basin and the two correlations.

    python locate_storms.py --forcings <CLDNet>/data/forcings --out ../../data/storms/storm_periods.csv

Needs numpy, pandas, scipy, xarray, cfgrib, pyproj and internet access (about 600 hourly files of
0.3 to 1.2 MB from the Iowa Environmental Mesonet archive, kept in --cache). About 10 minutes.
"""
import argparse
import glob
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import xarray as xr
from scipy.ndimage import map_coordinates
from scipy.signal import correlate

URL = "https://mesonet.agron.iastate.edu/archive/data/{t:%Y/%m/%d}/stage4/ST4.{t:%Y%m%d%H}.01h.grib"
# Stage IV lives on the HRAP polar-stereographic grid (4.7625 km at 60 N).
HRAP = pyproj.CRS.from_proj4("+proj=stere +lat_0=90 +lat_ts=60 +lon_0=-105 +x_0=0 +y_0=0 +R=6371200 +units=m +no_defs")
LL2H = pyproj.Transformer.from_crs("EPSG:4326", HRAP, always_xy=True)
A2H = pyproj.Transformer.from_crs("EPSG:5070", HRAP, always_xy=True)
A2LL = pyproj.Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)

NY, NX, DX = 39, 13, 4000.0                      # rain grid of the dataset
# Search grid in EPSG:5070 (4 km, south to north): the US east of the Rockies.
GX0, GY0, GNX, GNY = -1400e3, 500e3, 900, 700
EVENT_2013 = "2013-04-17_2013-04-21"             # the only folder not named by a number; trajectory 120
# Names for the places found (centre of the matched 156 km x 52 km field), assigned to the nearest centre.
SITES = [(42.20, -89.96, "NW Illinois, 65 km east of Dubuque IA"), (38.15, -87.71, "SW Indiana, Evansville"),
         (40.99, -92.36, "SE Iowa, Ottumwa"), (39.81, -95.01, "NE Kansas, near St. Joseph MO"),
         (43.95, -82.87, "Michigan Thumb"), (44.23, -92.21, "SE Minnesota, near Rochester"),
         (38.50, -94.54, "W Missouri, Kansas state line south of Kansas City"), (40.83, -96.62, "SE Nebraska, Lincoln"),
         (46.63, -102.66, "SW North Dakota, near Dickinson"), (38.82, -82.78, "S Ohio, near Portsmouth"),
         (43.21, -96.87, "SE South Dakota, near Sioux Falls"), (43.06, -89.57, "S Wisconsin, Madison"),
         (37.52, -88.37, "S Illinois, near Paducah KY"), (38.54, -85.43, "SE Indiana, Ohio River near Madison IN"),
         (43.44, -91.74, "NE Iowa, near La Crosse WI"), (37.71, -95.73, "SE Kansas, near Coffeyville"),
         (42.45, -85.91, "SW Michigan, near Kalamazoo"), (36.62, -89.52, "SE Missouri, near New Madrid"),
         (46.64, -96.98, "E North Dakota, near Fargo"), (41.67, -80.77, "NE Ohio, near Youngstown"),
         (42.02, -87.98, "Des Plaines basin")]


def km(lon1, lat1, lon2, lat2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def read_stage4(t, cache):
    """(mm in the hour ENDING at t UTC, HRAP grid origin and spacing, mask of cells without data); None when the
    archive has no file. Cells without data hold zero in the first array."""
    t = pd.Timestamp(t)
    f = cache / f"ST4.{t:%Y%m%d%H}.01h.grib"
    npz, miss = f.with_suffix(".v2.npz"), f.with_suffix(".missing")
    if npz.exists():
        d = np.load(npz)
        return d["a"], tuple(d["g"]), d["m"]
    if miss.exists():
        return None
    cache.mkdir(parents=True, exist_ok=True)
    if not f.exists():
        for k in range(4):
            try:
                with urllib.request.urlopen(URL.format(t=t), timeout=60) as r:
                    data = r.read()
                f.write_bytes(data)
                time.sleep(0.3)
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    miss.write_text("")
                    return None
                time.sleep(3 * (k + 1))
            except Exception:
                time.sleep(3 * (k + 1))
        else:
            return None
    ds = xr.open_dataset(f, engine="cfgrib", backend_kwargs={"indexpath": ""})
    a = np.asarray(ds[list(ds.data_vars)[0]].values, dtype=np.float32)
    lat, lon = np.asarray(ds.latitude.values, float), np.asarray(ds.longitude.values, float)
    ds.close()
    m = ~np.isfinite(a) | (a < 0) | (a > 2000)
    a[m] = 0.0
    x, y = LL2H.transform(np.where(lon > 180, lon - 360, lon), lat)
    g = (x[0, 0], y[0, 0], (x[0, -1] - x[0, 0]) / (x.shape[1] - 1), (y[-1, 0] - y[0, 0]) / (y.shape[0] - 1))
    np.savez_compressed(npz, a=a, g=np.array(g), m=m)
    return a, g, m


def sample(field, X, Y, missing=False):
    """Bilinear values of one Stage IV field at EPSG:5070 points (or, with missing=True, the no-data mask there)."""
    a, g, m = field
    hx, hy = A2H.transform(X, Y)
    rc = [(hy - g[1]) / g[3], (hx - g[0]) / g[2]]
    if missing:
        return map_coordinates(m.astype(np.float32), rc, order=0, mode="constant", cval=1.0)
    return map_coordinates(a, rc, order=1, mode="constant", cval=0.0)


def window(x0, y0):
    return np.meshgrid(x0 + DX * np.arange(NX), y0 + DX * np.arange(NY))


def corr(a, b):
    a, b = a.ravel(), b.ravel()
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0


def refine(fields, T, x0, y0, span):
    """Best correlation over sub-cell shifts of the window origin (500 m steps), over the hours for which the
    archive has data at that place. Returns (correlation, x0, y0, hours used)."""
    use = [i for i, f in enumerate(fields) if sample(f, *window(x0, y0), missing=True).mean() < 0.5]
    if not use:
        return (-2.0, x0, y0, 0)
    best = (-2.0, x0, y0, len(use))
    for dx in np.arange(-span, span + 1, 500.0):
        for dy in np.arange(-span, span + 1, 500.0):
            S = np.array([sample(fields[i], *window(x0 + dx, y0 + dy)) for i in use])
            c = corr(S, T[use])
            if c > best[0]:
                best = (c, x0 + dx, y0 + dy, len(use))
    return best


def wettest(p, t0, n, cache):
    steps = sorted(np.argsort(p.mean((1, 2)))[::-1][:n].tolist())
    got = [(k, read_stage4(t0 + pd.Timedelta(hours=k), cache)) for k in steps]
    got = [(k, f) for k, f in got if f is not None]
    if not got:
        raise SystemExit(f"no Stage IV file could be read for the storm starting {t0}")
    # The files store the northern row first although their y coordinate ascends (checked on the 2013 event, the
    # one storm known to be on the basin: correlation with the archive 0.996 north-first, 0.06 as labelled).
    return [f for _, f in got], np.array([p[k][::-1] for k, _ in got])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forcings", required=True, help="CLDNet data/forcings directory")
    ap.add_argument("--out", default="storm_periods.csv")
    ap.add_argument("--cache", default=str(Path(__file__).resolve().parent / "stage4_cache"))
    ap.add_argument("--hours", type=int, default=6, help="wettest hours compared per storm")
    ap.add_argument("--only", nargs="*", help="event folder ids to process (default: all)")
    args = ap.parse_args()
    cache = Path(args.cache)
    GXX, GYY = np.meshgrid(GX0 + DX * np.arange(GNX), GY0 + DX * np.arange(GNY))
    ones = np.ones((NY, NX))

    rows, sites = [], []
    names = sorted((os.path.basename(d)[6:] for d in glob.glob(os.path.join(args.forcings, "event_*"))),
                   key=lambda k: int(k) if k.isdigit() else 10 ** 6)
    for name in names:
        if args.only and name not in args.only:
            continue
        ds = xr.open_dataset(os.path.join(args.forcings, f"event_{name}", f"event_{name}.nc"))
        p, t = ds.prcp.values, ds.time.values
        xb, yb = float(ds.x.values[0]), float(ds.y.values[0])
        ds.close()
        t0 = pd.Timestamp(t[0])
        fields, T = wettest(p, t0, args.hours, cache)
        row = dict(traj_id=int(name) if name.isdigit() else 120, cldnet_event=name,
                   start_utc=f"{t0:%Y-%m-%d %H:%M}", end_utc=f"{pd.Timestamp(t[-1]):%Y-%m-%d %H:%M}",
                   mean_depth_mm=round(float(p[:96].sum(0).mean()), 1), peak_mm_per_h=round(float(p.max()), 1),
                   hours_requested=len(fields))
        # 1. at the basin, with the file's own coordinates
        row["corr_at_basin"] = round(corr(np.array([sample(f, *window(xb, yb)) for f in fields]), T), 3)
        # 2. everywhere: cosine similarity of the storm with every 39 x 13 window of the archive, summed over the hours
        G = [sample(f, GXX, GYY) for f in fields]
        num = sum(correlate(Gk, Tk, mode="valid", method="fft") for Gk, Tk in zip(G, T))
        den = sum(correlate(Gk * Gk, ones, mode="valid", method="fft") for Gk in G)
        t2 = float((T * T).sum())
        cs = num / np.sqrt(np.maximum(den, 0.25 * t2) * t2)      # windows much drier than the storm cannot win
        iy, ix = np.unravel_index(np.argmax(cs), cs.shape)
        c, x0, y0, used = refine(fields, T, GX0 + DX * ix, GY0 + DX * iy, 4000.0)
        if c < 0.93 and sites:
            # weak match (the archive has gaps at some hours): try the places already found, with twice the hours
            fields, T = wettest(p, t0, 2 * args.hours, cache)
            row["hours_requested"] = len(fields)
            c, x0, y0, used = max((refine(fields, T, sx, sy, 2000.0) for sx, sy in sites), key=lambda z: z[0])
        elif c >= 0.93 and not any(abs(sx - x0) < 6000 and abs(sy - y0) < 6000 for sx, sy in sites):
            sites.append((x0, y0))
        lon, lat = A2LL.transform(x0 + DX * (NX - 1) / 2, y0 + DX * (NY - 1) / 2)
        d = [km(lon, lat, s[1], s[0]) for s in SITES]
        row.update(corr_with_stage4=round(c, 3), field_centre_lat=round(float(lat), 2), field_centre_lon=round(float(lon), 2),
                   km_from_basin=int(round(km(lon, lat, SITES[-1][1], SITES[-1][0]))),
                   site=SITES[int(np.argmin(d))][2] if min(d) < 15 else "unnamed")
        row["on_basin"] = "yes" if row["corr_at_basin"] > 0.9 else "no"
        row["hours_compared"], row["hours_without_archive_data"] = used, row.pop("hours_requested") - used
        rows.append(row)
        print(f"{name:>22s} {row['start_utc']}  basin {row['corr_at_basin']:6.3f}  best {c:.3f} at "
              f"({lat:.2f}, {lon:.2f})  {row['km_from_basin']:5d} km  {row['site']}", flush=True)

    cols = ["traj_id", "cldnet_event", "start_utc", "end_utc", "on_basin", "site", "field_centre_lat",
            "field_centre_lon", "km_from_basin", "mean_depth_mm", "peak_mm_per_h", "corr_with_stage4", "corr_at_basin",
            "hours_compared", "hours_without_archive_data"]
    pd.DataFrame(rows).sort_values("traj_id")[cols].to_csv(args.out, index=False)
    print("wrote", args.out, len(rows), "rows")


if __name__ == "__main__":
    main()
