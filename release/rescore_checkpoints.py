#!/usr/bin/env python3
"""Rescore released models on prepared truth, in bounded spatial chunks."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from cstride.chicago import (load_chicago_manifest,load_chicago_normalizers,load_sensor_map_file,build_sensor_sequences,make_windows,add_chicago_input_noise,apply_chicago_sensor_dropout,get_chicago_state_array,chicago_include_precipitation_input,chicago_use_terrain_conditioning)
from cstride.chicago_forecast import build_chicago_feature_coords
from cstride.released import encode,decode_chunks
from cstride.models import CSTRIDE
from cstride.runtime import load_state_dict


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--models",nargs="+",default=[p.name for p in (ROOT/"checkpoints").iterdir() if p.is_dir()])
    parser.add_argument("--storms",nargs="+",type=int,default=[107,108,109,120])
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-size",type=int,default=100000)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/scores/models.json")
    args=parser.parse_args()
    if args.chunk_size<1:parser.error("chunk-size must be positive")
    torch.set_num_threads(4)
    data,_,entries=load_chicago_manifest(args.data_root)
    norms=load_chicago_normalizers(ROOT/"data/normalization/normalizers.npz")
    results={"seed":args.seed,"noise_note":"Source evaluation did not set a noise RNG seed; this audit uses an explicit seed.","models":{}}
    for name in args.models:
        run=ROOT/"checkpoints"/name
        config=yaml.safe_load((run/"config.yaml").read_text())
        config["basics"]["sensor_dropout_always_available_sensor_path"]=str(ROOT/"data/gauges/USGS_4sensors.json")
        sensors=load_sensor_map_file(run/"sensor_indices.json")
        rain=chicago_include_precipitation_input(config);terrain=chicago_use_terrain_conditioning(config)
        sequence=torch.stack(build_sensor_sequences(entries,data,sensors,norms,[0],rain))
        torch.manual_seed(args.seed)
        sequence=add_chicago_input_noise(sequence,config["basics"].get("noise",0),config["basics"]["num_sensors"])
        sequence,dropout=apply_chicago_sensor_dropout(sequence,config,config["basics"]["num_sensors"],1,sensors,entries,data)
        model=CSTRIDE(input_size=sequence.shape[-1],output_size=1,coord_size=5 if terrain else 2,device=torch.device(args.device),**config["model"]).to(args.device)
        model.load_state_dict(load_state_dict(run/"best_model.ckpt",args.device));model.eval()
        if dropout:
            dropout = {k: ("data/gauges/USGS_4sensors.json" if isinstance(v, str) and v == str(ROOT/"data/gauges/USGS_4sensors.json") else v) for k,v in dropout.items()}
        record={"storms":{},"dropout":dropout,"checkpoint":"best_model.ckpt"}
        for storm in args.storms:
            pos=next(i for i,e in enumerate(entries) if e["traj_id"]==storm);entry=entries[pos]
            with np.load(ROOT/f"data/grids/grid_traj{storm}.npz",allow_pickle=False) as grid:
                coords=build_chicago_feature_coords(grid["coords"],grid["elevation"],grid["manning"],grid["slope"],norms,terrain)
            truth=get_chicago_state_array(data,entry)
            lag=config["basics"]["lag"];start=max(lag,24) if name.startswith("lag") else lag
            curves=[];rmses=[];counts=np.zeros(3,dtype=np.int64);started=time.perf_counter()
            with torch.inference_mode():
                for t in range(start,96):
                    latent=encode(model,sequence[pos:pos+1,t-lag:t].to(args.device))
                    sq=den=0.
                    for offset,values in decode_chunks(model,latent,coords,args.chunk_size):
                        pred=norms["depth"].inverse_transform(values)
                        target=np.asarray(truth[t,offset:offset+len(pred),0],dtype=np.float32)
                        sq+=np.sum((pred-target)**2,dtype=np.float64);den+=np.sum(target**2,dtype=np.float64)
                        a,b=pred>=.5,target>=.5
                        counts+=np.array([np.count_nonzero(a&b),np.count_nonzero(a&~b),np.count_nonzero(~a&b)])
                    curves.append(float(np.sqrt(sq/den)));rmses.append(float(np.sqrt(sq/len(coords))))
            record["storms"][str(storm)]={"target_start":start,"mean_relative_error":float(np.mean(curves)),"mean_rmse_m":float(np.mean(rmses)),"relative_error":curves,"rmse_m":rmses,"inundation_counts":counts.tolist(),"seconds_including_scoring":time.perf_counter()-started}
            print(name,storm,record["storms"][str(storm)]["mean_relative_error"],flush=True)
        record["four_event_mean_relative_error"]=float(np.mean([x["mean_relative_error"] for x in record["storms"].values()]))
        record["four_event_mean_rmse_m"]=float(np.mean([x["mean_rmse_m"] for x in record["storms"].values()]))
        results["models"][name]=record
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(results,indent=2)+"\n")
        del model
        if str(args.device).startswith("cuda"):torch.cuda.empty_cache()


if __name__=="__main__":main()
