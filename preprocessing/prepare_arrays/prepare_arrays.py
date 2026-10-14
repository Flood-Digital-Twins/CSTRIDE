#!/usr/bin/env python3
"""Convert SynxFlow collected fields to the historical C-STRIDE prepared layout.

The legacy Manning order is explicit because released models were trained on it.
All flow arrays are written to the caller's output directory, never bundled inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def raw_static(directory, legacy_manning):
    dem = np.load(directory/"DEM.npy",mmap_mode="r")
    elevation = np.asarray(dem[2] if dem.ndim==3 else dem,dtype=np.float32)
    valid = np.isfinite(elevation)
    mean,std = float(np.nanmean(elevation)),float(np.nanstd(elevation))
    filled = np.nan_to_num(elevation,nan=mean)
    gy,gx = np.gradient(filled)
    values=[]
    with (directory/"input/field/manning.dat").open() as stream:
        for line in stream:
            if line.startswith("$Boundary"): break
            if line.startswith("$"): continue
            fields=line.split()
            if len(fields)>=2:
                try: values.append(float(fields[1]))
                except ValueError: pass
    values=np.asarray(values,dtype=np.float32)
    if len(values)!=np.count_nonzero(valid):
        raise ValueError("Manning record count does not match finite DEM cells")
    manning=np.full(elevation.shape,np.nan,dtype=np.float32)
    if legacy_manning:
        manning.ravel()[valid.ravel()]=values
    else:
        flipped=np.flipud(manning).copy()
        flipped.ravel()[np.flipud(valid).ravel()]=values
        manning=np.flipud(flipped)
    return elevation,valid,mean,std,gx,gy,manning


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir",type=Path,required=True,help="SynxFlow event directory with flow_variables.npy, DEM.npy and input/field/manning.dat")
    parser.add_argument("--rain-source",type=Path,help="Collected rain_source.npy; defaults to event directory")
    parser.add_argument("--traj-id",type=int,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--burn-in",type=int,default=1)
    parser.add_argument("--time-steps",type=int,default=96)
    parser.add_argument("--depth-threshold",type=float,default=0.1)
    parser.add_argument("--static-precision", choices=("historical", "raw-float32", "quantized-float16"), default="historical", help="Historical: raw static values for storm 120, quantized values for numbered storms")
    parser.add_argument("--legacy-manning-order",action="store_true",help="Required to recreate the features used by released checkpoints")
    parser.add_argument("--reference-root",type=Path,help="Compare reconstructed arrays with an existing prepared directory")
    parser.add_argument("--indices-only", action="store_true", help="Write exact event mask cell/row/column indices only")
    args=parser.parse_args()
    flow=np.load(args.input_dir/"flow_variables.npy",mmap_mode="r")
    if flow.ndim != 4: raise ValueError("Expected (T,3,H,W) or (T,H,W,3)")
    channels_first=flow.shape[1]==3
    height,width=flow.shape[2:] if channels_first else flow.shape[1:3]
    def frame(t):
        value=flow[t]
        return np.moveaxis(value,0,-1) if channels_first else value
    if args.burn_in<0 or args.time_steps<1 or args.burn_in+args.time_steps>len(flow):
        raise ValueError("Requested time range is outside collected flow")
    elev,valid,mean,std,gx,gy,manning=raw_static(args.input_dir,args.legacy_manning_order)
    if elev.shape!=(height,width): raise ValueError("DEM and flow dimensions disagree")
    peak=np.full((height,width),-np.inf,dtype=np.float32)
    for t in range(args.burn_in,args.burn_in+args.time_steps):
        depth=np.asarray(frame(t)[...,0],dtype=np.float32)
        valid &= np.isfinite(depth)
        peak=np.maximum(peak,depth)
    valid &= peak>=args.depth_threshold
    cells=np.flatnonzero(valid.ravel());rows,cols=np.unravel_index(cells,(height,width))
    x=2*(np.arange(width,dtype=np.float32)/width-.5)
    y=2*(np.arange(height,dtype=np.float32)/height-.5)*(height/float(width))
    coords=np.column_stack([x[cols],y[rows]]).astype(np.float16)
    z=((elev-mean)/std).astype(np.float16)
    # Preserve the quantized training elevation/Manning representation.
    physical=z.astype(np.float32)*std+mean
    roughness=(manning*100).astype(np.float16).astype(np.float32)/100
    slope=np.sqrt(gx**2+gy**2).astype(np.float16).astype(np.float32)
    if args.static_precision == "raw-float32" or args.static_precision == "historical" and args.traj_id == 120:
        physical=elev
        slope=np.sqrt(gx**2+gy**2).astype(np.float32)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output_dir/f"grid_indices_traj{args.traj_id}.npz",cell_indices=cells,rows=rows,cols=cols,height=height,width=width)
    if args.indices_only:
        args.output_dir.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.output_dir/f"grid_indices_traj{args.traj_id}.npz",cell_indices=cells,rows=rows,cols=cols,height=height,width=width)
        print(f"Recovered {len(cells)} raster indices for {args.traj_id}",flush=True)
        return
    names={key:f"{key}_traj{args.traj_id}.npy" for key in ("state","depth","coords","elevation","manning","slope","precip")}
    state=np.lib.format.open_memmap(args.output_dir/names["state"],mode="w+",dtype=np.float16,shape=(1,args.time_steps,len(cells),3))
    depth=np.lib.format.open_memmap(args.output_dir/names["depth"],mode="w+",dtype=np.float16,shape=(args.time_steps,len(cells),1))
    for step,t in enumerate(range(args.burn_in,args.burn_in+args.time_steps)):
        value=frame(t).reshape(-1,3)[cells].astype(np.float16)
        state[0,step]=value;depth[step]=value[:,:1]
    state.flush();depth.flush()
    for key,value in (("coords",coords),("elevation",physical.ravel()[cells]),("manning",roughness.ravel()[cells]),("slope",slope.ravel()[cells])):
        np.save(args.output_dir/names[key],value)
    rain=np.load(args.rain_source or args.input_dir/"rain_source.npy",mmap_mode="r")
    if rain.ndim>=3 and rain.shape[0]==1: rain=rain[0]
    rain=np.asarray(rain[:args.time_steps],dtype=np.float32).reshape(args.time_steps,-1).astype(np.float16)
    np.save(args.output_dir/names["precip"],rain)
    manifest_path=args.output_dir/"manifest.json"
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {"time_steps":args.time_steps,"files":[]}
    entry=dict(traj_id=args.traj_id,**{k+"_file":v for k,v in names.items()})
    manifest["files"]=[e for e in manifest["files"] if e["traj_id"]!=args.traj_id]+[entry]
    manifest["files"].sort(key=lambda e:e["traj_id"])
    manifest["preparation"]={"burn_in":args.burn_in,"depth_threshold":args.depth_threshold,"legacy_manning_order":args.legacy_manning_order,"dem_mean":mean,"dem_std":std}
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
    report={"storm":args.traj_id,"cells":len(cells),"comparisons":{}}
    if args.reference_root:
        for key,name in names.items():
            old=np.load(args.reference_root/name,mmap_mode="r");new=np.load(args.output_dir/name,mmap_mode="r")
            equal=old.shape==new.shape and old.dtype==new.dtype
            maximum=0.
            if equal:
                for index in range(old.shape[0]):
                    a,b=old[index],new[index]
                    equal=equal and np.array_equal(a,b,equal_nan=True)
                    maximum=max(maximum,float(np.nanmax(np.abs(a.astype(np.float32)-b.astype(np.float32)))))
            report["comparisons"][key]={"exact":bool(equal),"reference_shape":list(old.shape),"reconstructed_shape":list(new.shape),"max_absolute_difference":maximum if old.shape==new.shape else None}
    (args.output_dir/f"verification_traj{args.traj_id}.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)
    if args.reference_root and not all(r["exact"] for r in report["comparisons"].values()): raise SystemExit(1)


if __name__=="__main__":main()
