#!/usr/bin/env python3
"""Measure synchronized next-step decoding from the compact release inputs."""
import argparse,json,sys,time,subprocess,tempfile,shutil,re
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from cstride.released import load_inputs,encode
from cstride.chicago import load_sensor_map_file


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",default="full")
    parser.add_argument("--storm",type=int,default=120)
    parser.add_argument("--cldnet-root",type=Path)
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-size",type=int,default=100000)
    parser.add_argument("--repeats",type=int,default=1)
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/timing.json")
    parser.add_argument("--forecast-data-root",type=Path,help="Also time 40 gauge forecasts of 24 steps using prepared simulation truth")
    args=parser.parse_args()
    if args.chunk_size<1 or args.repeats<1:parser.error("chunk-size and repeats must be positive")
    model,config,norms,seq,coords,_,_=load_inputs(args.model,args.storm,args.cldnet_root,device=args.device)
    lag=config["basics"]["lag"]
    sensor=load_sensor_map_file(ROOT/f"checkpoints/{args.model}/sensor_indices.json")[args.storm]
    def sync():
        if str(args.device).startswith("cuda"):torch.cuda.synchronize(args.device)
    timings={}
    with torch.inference_mode():
        history=seq[:lag].unsqueeze(0).to(args.device)
        latent=encode(model,history);model.decode(torch.tensor(coords[:1024],device=args.device).unsqueeze(0),latent);sync()
        for mode,query in (("full_fields",coords),("gauges",coords[sensor])):
            query=torch.tensor(query,device=args.device).unsqueeze(0)
            elapsed=[]
            for _ in range(args.repeats):
                sync();start=time.perf_counter()
                for t in range(lag,len(seq)):
                    latent=encode(model,seq[t-lag:t].unsqueeze(0).to(args.device))
                    # Concatenation matches the paper's next-step timing scope.
                    torch.cat([model.decode(query[:,i:i+args.chunk_size],latent) for i in range(0,query.shape[1],args.chunk_size)],dim=1)
                sync();elapsed.append(time.perf_counter()-start)
            timings[mode]={"seconds":elapsed,"mean_seconds_per_snapshot":float(np.mean(elapsed)/(len(seq)-lag)),"query_points":query.shape[1]}
    report={"model":args.model,"storm":args.storm,"device":torch.cuda.get_device_name(args.device) if str(args.device).startswith("cuda") else "CPU","target_snapshots":len(seq)-lag,"scope":"Warm synchronized encoder, chunked decoder, input transfer and concatenation; excludes normalization, CPU scoring, writing, and input preparation.","timings":timings,"paper_reference":{"full_fields_seconds":21.99,"gauge_ms_per_snapshot":3.55,"40_forecasts_24steps_seconds_including_preparation_and_scoring":247.37},"forecast_timing_command":"forecast_observations.py reports preparation/scoring-inclusive runtime from the prepared-data workflow; its scope differs from these next-step timings."}
    if args.forecast_data_root:
        with tempfile.TemporaryDirectory(prefix="cstride-timing-") as temporary:
            run=Path(temporary)
            for asset in (ROOT/"checkpoints"/args.model).iterdir():
                if asset.is_file():shutil.copyfile(asset,run/asset.name)
            command=[sys.executable,str(ROOT/"scripts/forecast_observations.py"),"--path",str(run),"--data-root",str(args.forecast_data_root.resolve()),"--normalizers",str(ROOT/"data/normalization/normalizers.npz"),"--horizon","24","--num_starts","10","--eval_seed","0","--obs_include_precipitation_input","--device",args.device]
            completed=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
            match=re.search(r"Inference time: ([0-9.]+) seconds",(run/"results.txt").read_text())
            if not match:raise RuntimeError("Forecast evaluator did not report inference timing")
            report["forecast"]={"seconds_including_preparation_and_scoring":float(match.group(1)),"samples":40,"horizon":24,"evaluation_seed":0,"scope":"Original prepared-data evaluator; preparation and scoring included, existing gauge forecaster loaded."}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))


if __name__=="__main__":main()
