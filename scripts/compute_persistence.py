import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

"""Persistence on precisely the 40 documented forecast origins (no model fitting)."""
import json
import argparse
import os
import numpy as np
from generate_paper_results import OUT, RUN_ROOT, RUNS
from plot_chicago_future_precipitation import select_section
from plot_chicago_precipitation_mismatch import parse_evaluation_starts
from cstride.chicago import load_chicago_manifest, get_chicago_state_array

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--data-root",required=True);args=p.parse_args()
    os.environ["CSTRIDE_DATA_ROOT"]=args.data_root
    OUT.mkdir(parents=True,exist_ok=True)
    root, manifest, entries=load_chicago_manifest()
    section=select_section(RUN_ROOT/'full',True,0.,True)
    starts=parse_evaluation_starts(section, entries)
    assert [entries[i]['traj_id'] for i in starts]==[107,108,109,120]
    errors=[]; origins=[]
    for i,origins_for_event in starts.items():
        state=get_chicago_state_array(root,entries[i])
        for s in origins_for_event:
            previous=np.asarray(state[s-1,:,0],dtype=np.float32); curve=[]
            for t in range(s,s+24):
                truth=np.asarray(state[t,:,0],dtype=np.float32)
                curve.append(float(np.sqrt(np.sum((previous-truth)**2,dtype=np.float64)/np.sum(truth**2,dtype=np.float64))))
            errors.append(curve);origins.append(dict(trajectory_id=entries[i]['traj_id'],first_target=s))
    errors=np.asarray(errors)
    result=dict(metadata=section['metadata'],means={i+1:v for i,v in enumerate(errors.mean(0))},
        stds={i+1:v for i,v in enumerate(errors.std(0))},origins=origins,
        definition='Persist true complete field at first_target-1; privileged full-state reference, not a sparse-gauge deployable baseline')
    (OUT/'persistence.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(OUT/'persistence_rollouts.npz',relative_error=errors)
    print('Persistence leads 1,4,6,12,24:',errors.mean(0)[[0,3,5,11,23]],flush=True)


if __name__ == "__main__":
    main()
