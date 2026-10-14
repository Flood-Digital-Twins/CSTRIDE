import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

"""Generate the paper prediction and diagnostic products from released checkpoints.

Prepared simulation arrays are supplied by the caller. Outputs identify models
by label and use the published retrospective USGS alignment.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from cstride.chicago import (load_sensor_map_file, load_chicago_normalizers,MinMaxNormalizer, build_sensor_sequences, chicago_split_indices,
    fit_chicago_normalizers, get_chicago_state_array, load_chicago_manifest,
    load_chicago_sensors, make_windows, chicago_include_precipitation_input,
    chicago_use_terrain_conditioning)
from score_additional_metrics import (build_feature_coords_for_entry,
    encode_latent, load_state_dict_compatible, nash_sutcliffe_efficiency,
    kling_gupta_efficiency)
from cstride.models import CSTRIDE

OUT = ROOT / 'outputs/paper'
RUN_ROOT = ROOT / 'checkpoints'
RUNS = ('full', 'vanilla', 'terrain', 'forcing', 'lag1', 'lag4', 'lag6', 'lag24', 'four_gauges')
EVENTS = [107,108,109,120]
STATIONS = ['05527800','05528000','05529000','05531500','05532500','05540130']
NAMES = ['Russell','Gurnee','Des Plaines','Salt Creek','Riverside','DuPage']


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def projected(coords):
    # EPSG:5070 affine conversion for the paper diagnostic coordinates.
    col=(coords[:,0]/2+0.5)*1661
    row=(coords[:,1]/(2*(5075/1661))+0.5)*5075
    return np.column_stack(((635308.3453469264+col*30)/1000,
                            (2217115.724870439-row*30)/1000))


def representatives(depth, xy):
    # Use the alternating-row diagnostic selection shown in the paper. It does not use
    # four northing bands as the previous manuscript claimed.
    peak=np.max(depth[12:],axis=0)
    tv=np.sum(np.abs(np.diff(depth[12:],axis=0)),axis=0)
    northing=xy[:,1]; rows=np.round(northing,6); unique=np.unique(rows);used=set();reps=[]
    for k,frac in enumerate(np.arange(.26,.98,.09)):
        target=northing.max()-frac*np.ptp(northing)
        available=np.asarray([r for r in unique if r not in used])
        row=available[np.argmin(abs(available-target))]
        candidates=np.flatnonzero(np.isclose(rows,row)); metric=peak if k%2==0 else tv
        top=candidates[np.isclose(metric[candidates],metric[candidates].max())]
        idx=int(top[np.argmin(abs(xy[top,0]-xy[candidates,0].mean()))])
        reps.append(dict(label=chr(65+k),idx=idx,kind='peak' if k%2==0 else 'tv',
                         peak_m=float(peak[idx]),tv_m=float(tv[idx])))
        used.add(row)
    return reps


def observations(root, sim_depth, elevation):
    times=pd.date_range('2013-04-17 08:00',periods=96,freq='h')
    columns=[]; provenance=[]
    for station in STATIONS:
        matches=[]
        for path in (ROOT/'data/gauges/usgs_2013/raw_stage').glob('*_height.csv'):
            d=pd.read_csv(path,dtype={'site_no':str})
            if len(d) and d.site_no.iloc[0].zfill(8)==station:matches.append((path,d))
        assert len(matches)==1,(station,len(matches))
        path,d=matches[0]; datum=float(d.alt_va.iloc[0]); manual=10. if int(path.name.split('_')[0]) in (14,20) else 0.
        d['datetime']=pd.to_datetime(d.datetime)
        d['wse']=.3048*(pd.to_numeric(d['Gage Height (ft)'],errors='coerce')+datum+manual)
        d=d.dropna(subset=['wse']).drop_duplicates('datetime',keep='last').set_index('datetime').sort_index()
        d=d.loc[times[0]:times[-1],'wse'].resample('h').nearest().reindex(times)
        missing=int(d.isna().sum()); d=d.interpolate(method='nearest',limit_direction='both').ffill().bfill()
        assert d.notna().all()
        columns.append(d.to_numpy())
        provenance.append(dict(station=station,source=str(path),source_sha256=sha(path),datum_ft=datum,
            datum=str(pd.read_csv(path).alt_datum_cd.iloc[0]),manual_addition_ft=manual,filled_hour_count=missing))
    obs=np.column_stack(columns)
    shift=obs[12:].mean(0)-(sim_depth[12:]+elevation).mean(0)
    input_depth=np.maximum(obs-shift-elevation,0)
    np.savez_compressed(OUT/'usgs_observations.npz',obs_wse=obs,shift_m=shift,
        input_depth=input_depth,elevation=elevation,times=times.astype(str).to_numpy(dtype=str),sim_depth=sim_depth)
    (OUT/'usgs_preprocessing.json').write_text(json.dumps(dict(stations=provenance,
        timeline='Notebook convention: hourly 2013-04-17 08:00 through 2013-04-21 07:00; hourly source interval and one-frame burn-in documented; timezone unverified',
        offset='Added to both simulator and predictions: mean(USGS WSE[12:96])-mean(simulated WSE[12:96]); predictions not separately centered',
        input_conversion='max(USGS WSE-offset-ground elevation,0); full-window retrospective alignment',
        peak_timing='first occurrence of maximum on common indices 12:96; prediction minus reference in assumed hours'),indent=2))
    return input_depth


def load_normalizers(root,entries):
    cache=ROOT/'data/normalization/normalizers.npz'
    if cache.exists():
        z=np.load(cache); return {k:MinMaxNormalizer(z[k+'_min'],z[k+'_max']) for k in ['state','depth','precip','elevation','manning','slope']}
    print('Fitting normalizers on training data',flush=True)
    idx,_,_=chicago_split_indices(entries); n=fit_chicago_normalizers(entries,idx,root)
    np.savez(cache,**{k+s:v for k,norm in n.items() for s,v in [('_min',norm.min_value),('_max',norm.max_value)]})
    (ROOT/'data/normalization/normalization_provenance.json').write_text(json.dumps(dict(training_events=90,test_events=4),indent=2))
    return n


def run(args):
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    root,manifest,entries=load_chicago_manifest(args.data_root); norms=load_normalizers(root,entries)
    if (OUT/'reference_traj120.npz').exists():
        ref=np.load(OUT/'reference_traj120.npz');gidx=ref['gauge_indices'];points=ref['point_indices']
        snapshot=int(ref['snapshot_index']);input_depth=np.load(OUT/'usgs_observations.npz')['input_depth']
    else:
        entry120=next(e for e in entries if e['traj_id']==120)
        state120=get_chicago_state_array(root,entry120);depth120=np.asarray(state120[:,:,0],dtype=np.float32)
        xy=projected(np.load(root/entry120['coords_file']).astype(np.float32))
        gauge_map=load_sensor_map_file(ROOT/'data/gauges/USGS_6sensors.json');gidx=np.asarray(gauge_map[120])
        reps=representatives(depth120,xy);ridx=np.array([r['idx'] for r in reps]);points=np.r_[gidx,ridx]
        (OUT/'diagnostic_locations.json').write_text(json.dumps(reps,indent=2))
        elevation=np.load(root/entry120['elevation_file'])
        input_depth=observations(root,depth120[:,gidx],elevation[gidx])
        snapshot=int(np.argmax(np.sum(depth120[12:],axis=1)))+12
        np.savez_compressed(OUT/'reference_traj120.npz',xy=xy,elevation=elevation,gauge_indices=gidx,
            point_indices=points,point_depth=depth120[:,points],snapshot_index=snapshot,snapshot_depth=depth120[snapshot])
        del state120,depth120
    for label in args.runs:
        path=RUN_ROOT/label;config=yaml.safe_load((path/'config.yaml').read_text())
        precip=chicago_include_precipitation_input(config);terrain=chicago_use_terrain_conditioning(config)
        lag=int(config['basics']['lag']);smap=load_chicago_sensors(path,config)
        ckpt=path/'best_model.ckpt'; checkpoint_hash=sha(ckpt)
        model=CSTRIDE(input_size=len(smap[120])+507*precip,coord_size=5 if terrain else 2,
            output_size=1,device=args.device,**config['model']).to(args.device)
        model.load_state_dict(load_state_dict_compatible(ckpt,args.device));model.eval()
        event_ids=[120] if label=='four_gauges' else EVENTS
        for tid in event_ids:
            out=OUT/f'{label}_traj{tid}.npz'
            if out.exists():
                old=json.loads(out.with_suffix('.json').read_text())
                assert old['checkpoint_sha256']==checkpoint_hash and old['config_sha256']==sha(path/'config.yaml'), 'Cached run identity changed'
                print('Already generated',out,flush=True)
                continue
            entry=next(e for e in entries if e['traj_id']==tid)
            state=get_chicago_state_array(root,entry)
            seq=build_sensor_sequences([entry],root,smap,norms,[0],include_precipitation=precip)[0]
            windows=make_windows(seq,lag)
            features=build_feature_coords_for_entry(entry,root,norms,terrain)
            coords=torch.as_tensor(features,device=args.device).unsqueeze(0)
            start=max(lag,24) if label.startswith('lag') else lag
            ts=np.arange(start,96);errs=[];rmses=[];predpoints=[];counts=np.zeros(3,dtype=np.int64);snapshot_pred=None
            elapsed=0.
            with torch.inference_mode():
                # Warm up before synchronized forward-pass-only timing.
                latent=encode_latent(model,windows[:1].to(args.device));model.decode(coords[:,:1024],latent);torch.cuda.synchronize(args.device) if str(args.device).startswith('cuda') else None
                for t in ts:
                    torch.cuda.synchronize(args.device) if str(args.device).startswith('cuda') else None;tic=time.perf_counter()
                    latent=encode_latent(model,windows[t-lag:t-lag+1].to(args.device))
                    chunks=[model.decode(coords[:,s:s+args.chunk],latent) for s in range(0,coords.shape[1],args.chunk)]
                    pred=torch.cat(chunks,dim=1);torch.cuda.synchronize(args.device) if str(args.device).startswith('cuda') else None;elapsed+=time.perf_counter()-tic
                    pred=norms['depth'].inverse_transform(pred.cpu().numpy().reshape(-1))
                    truth=np.asarray(state[t,:,0],dtype=np.float32)
                    sq=np.square(pred-truth);errs.append(float(np.sqrt(np.sum(sq,dtype=np.float64)/np.sum(truth**2,dtype=np.float64))))
                    rmses.append(float(np.sqrt(sq.mean(dtype=np.float64))))
                    a=pred>=.5;b=truth>=.5;counts+=np.array([np.count_nonzero(a&b),np.count_nonzero(a&~b),np.count_nonzero(~a&b)])
                    if tid==120:
                        predpoints.append(pred[points])
                        if t==snapshot and label=='full':snapshot_pred=pred.copy()
                custom_pred=[]
                if tid==120 and label in ['full','four_gauges']:
                    positions=[int(np.flatnonzero(gidx==i)[0]) for i in smap[120]]
                    custom=seq.clone();custom[:,:len(positions)]=torch.tensor(norms['depth'].transform(input_depth[:,positions]))
                    custom_windows=make_windows(custom,lag);metric_coords=coords[:,gidx]
                    for w in custom_windows:
                        latent=encode_latent(model,w[None].to(args.device))
                        custom_pred.append(norms['depth'].inverse_transform(model.decode(metric_coords,latent).cpu().numpy().reshape(-1)))
            np.savez_compressed(out,timesteps=ts,relative_error=errs,rmse_m=rmses,counts=counts,
                point_predictions=np.asarray(predpoints),usgs_input_predictions=np.asarray(custom_pred),
                snapshot_prediction=np.asarray(snapshot_pred) if snapshot_pred is not None else np.array([]))
            meta=dict(model=label,model_directory=label,checkpoint_sha256=checkpoint_hash,config_sha256=sha(path/'config.yaml'),
                trajectory_id=tid,lag=lag,target_indices=ts.tolist(),evaluation_cells=int(coords.shape[1]),
                sensor_indices=np.asarray(smap[tid]).tolist(),terrain=terrain,precipitation=precip,
                mean_relative_error=float(np.mean(errs)),mean_rmse_m=float(np.mean(rmses)),
                forward_seconds=elapsed,device=torch.cuda.get_device_name(args.device) if str(args.device).startswith('cuda') else 'CPU',device_index=args.device,
                timing_scope='Synchronized encoder and chunked decoder plus input transfer and concatenation; excludes CPU scoring and inverse scaling')
            out.with_suffix('.json').write_text(json.dumps(meta,indent=2));print(label,tid,meta['mean_relative_error'],elapsed,flush=True)
            del coords
        del model;torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--chunk',type=int,default=100000)
    p.add_argument('--runs',nargs='+',choices=list(RUNS),default=list(RUNS));run(p.parse_args())
