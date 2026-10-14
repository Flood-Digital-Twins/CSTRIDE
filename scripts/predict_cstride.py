#!/usr/bin/env python3
"""Predict a released storm from compact inputs, with no simulated fields."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from cstride.chicago import chicago_include_precipitation_input
from cstride.released import load_inputs, encode, decode_chunks, load_gauge_forecaster


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="full", choices=[p.name for p in (ROOT/"checkpoints").iterdir() if p.is_dir()])
    parser.add_argument("--storm", type=int, choices=(107,108,109,120), required=True)
    parser.add_argument("--cldnet-root", type=Path, help="CLDNet checkout providing public rainfall")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-size", type=int, default=100000)
    parser.add_argument("--observations", choices=("simulated","usgs"), default="simulated")
    parser.add_argument("--start", type=int, help="First target time; defaults to lag")
    parser.add_argument("--steps", type=int, help="Number of target times; defaults to remaining event")
    parser.add_argument("--forecast", action="store_true", help="Roll out using the released gauge predictor")
    parser.add_argument("--obs-rainfall", action="store_true", help="Rainfall-aware auxiliary gauge predictor")
    parser.add_argument("--out-dir", type=Path, default=ROOT/"outputs/predictions")
    parser.add_argument("--save-fields", action="store_true", help="Write float32 depth fields to a .npy file")
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("chunk-size must be positive")
    model, config, norms, sequence, coords, rain, positions = load_inputs(args.model,args.storm,args.cldnet_root,args.observations,args.device,auxiliary_rainfall=args.obs_rainfall)
    lag = config["basics"]["lag"]
    start = lag if args.start is None else args.start
    steps = len(sequence)-start if args.steps is None else args.steps
    if start < lag or steps < 1 or start+steps > len(sequence):
        parser.error("Target range must start at/after lag and fit inside the event")
    predictor = metadata = None
    if args.forecast:
        predictor,metadata = load_gauge_forecaster(args.model,args.obs_rainfall,args.device)
        obs_history = sequence[start-lag:start,:len(positions)]
        if args.obs_rainfall:
            if rain is None:
                parser.error("Rainfall-aware auxiliary inference requires public rainfall inputs")
            obs_history = torch.cat([obs_history,torch.tensor(rain[start-lag:start])],dim=-1)
        if obs_history.shape[-1] != metadata["input_size"]:
            parser.error("Auxiliary checkpoint input size does not match the selected inputs")
        obs_history = obs_history.unsqueeze(0).to(args.device)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    stem = args.out_dir/f"{args.model}_traj{args.storm}_{'forecast' if args.forecast else 'next_step'}"
    fields = np.lib.format.open_memmap(str(stem)+"_fields.npy",mode="w+",dtype=np.float32,shape=(steps,len(coords))) if args.save_fields else None
    peak = np.full(len(coords),-np.inf,dtype=np.float32)
    history = sequence[start-lag:start].unsqueeze(0).to(args.device)
    with torch.inference_mode():
        for lead,t in enumerate(range(start,start+steps)):
            if not args.forecast:
                history = sequence[t-lag:t].unsqueeze(0).to(args.device)
            latent = encode(model,history)
            for offset,values in decode_chunks(model,latent,coords,args.chunk_size):
                depth = norms["depth"].inverse_transform(values)
                peak[offset:offset+len(depth)] = np.maximum(peak[offset:offset+len(depth)],depth)
                if fields is not None: fields[lead,offset:offset+len(depth)] = depth
            if predictor is not None:
                predicted = predictor(obs_history)
                forcing = torch.tensor(rain[t],device=args.device).unsqueeze(0) if rain is not None else None
                next_backbone = predicted if not chicago_include_precipitation_input(config) else torch.cat([predicted,forcing],dim=-1)
                next_observation = predicted if not args.obs_rainfall else torch.cat([predicted,forcing],dim=-1)
                history = torch.cat([history[:,1:],next_backbone.unsqueeze(1)],dim=1)
                obs_history = torch.cat([obs_history[:,1:],next_observation.unsqueeze(1)],dim=1)
    if fields is not None: fields.flush()
    np.savez_compressed(str(stem)+"_peak.npz",peak_depth_m=peak,coords=coords[:,:2],target_times=np.arange(start,start+steps))
    Path(str(stem)+".json").write_text(json.dumps(dict(model=args.model,storm=args.storm,start=start,steps=steps,forecast=args.forecast,observations=args.observations,cells=len(coords),clipping=False),indent=2)+"\n")
    print(f"Wrote {stem}_peak.npz ({len(coords):,} cells, {steps} target times)")


if __name__ == "__main__":
    main()
