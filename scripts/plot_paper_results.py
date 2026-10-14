import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

"""Regenerate the model figures used in the paper."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from generate_paper_results import OUT, RUNS, EVENTS, NAMES, STATIONS
from score_additional_metrics import nash_sutcliffe_efficiency, kling_gupta_efficiency

FIG=ROOT / 'figures';PREFIX=''
LABELS={'full':'C-STRIDE (full)','terrain':'C-STRIDE (terrain)','forcing':'C-STRIDE (forcing)','vanilla':'Vanilla STRIDE'}


def save(fig,name):
    fig.savefig(FIG/name,dpi=300,bbox_inches='tight');plt.close(fig);print('Saved',name,flush=True)


def curves(labels,name,start):
    fig,ax=plt.subplots(figsize=(6,3))
    for key,label in labels:
        z=[np.load(OUT/f'{key}_traj{t}.npz') for t in EVENTS]
        x=np.arange(start,96)
        values=np.stack([v['relative_error'][np.isin(v['timesteps'],x)] for v in z])
        assert values.shape==(4,len(x))
        mean=values.mean(0);std=values.std(0)
        ax.plot(x,mean,linewidth=2,label=label)
        ax.fill_between(x,mean-std,mean+std,alpha=.18)
    ax.set_ylim(0, .4 if start==12 else .35)
    ax.set_xlim(start,95);ax.set_xlabel('Original timestep');ax.set_ylabel('Mean relative $L_2$ depth error')
    ax.legend(fontsize=8);fig.tight_layout();save(fig,name)


def map_field(ax,ref,values,cmap='terrain',**kwargs):
    xy=ref['xy']; artist=ax.scatter(xy[:,0],xy[:,1],c=values,s=3,cmap=cmap,edgecolors='none',**kwargs)
    ax.set_aspect('equal');ax.set_xlabel('Easting (km)');ax.set_ylabel('Northing (km)')
    return artist


def gauges(ax,ref,label_position="top_right"):
    xy=ref['xy'][ref['gauge_indices']]
    ax.scatter(xy[:,0],xy[:,1],marker='^',c='red',s=110,edgecolors='white',zorder=4)
    offset=(6,-6) if label_position=='bottom_right' else (6,6)
    vertical='top' if label_position=='bottom_right' else 'bottom'
    for i,(x,y) in enumerate(xy):
        ax.annotate(str(i+1),(x,y),xytext=offset,textcoords='offset points',
                    color='red',fontsize=12,weight='bold',ha='left',va=vertical,zorder=5)


def format_time(ax,times):
    ax.set_xlim(times[0],times[-1]);ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator());ax.grid(alpha=.25)


def diagnostic(ref):
    reps=json.loads((OUT/'diagnostic_locations.json').read_text());obs=np.load(OUT/'usgs_observations.npz')
    times=pd.to_datetime(obs['times'][12:]);truth=ref['point_depth'][12:,6:]
    predictions={k:np.load(OUT/f'{k}_traj120.npz')['point_predictions'][:,6:] for k in ['full','vanilla','terrain']}
    fig=plt.figure(figsize=(12,8),constrained_layout=True);gs=GridSpec(4,3,figure=fig,width_ratios=[1.15,1.4,1.15])
    inner=gs[:,1].subgridspec(2,1,height_ratios=[4,.35]);ax=fig.add_subplot(inner[0]);leg=fig.add_subplot(inner[1]);leg.axis('off')
    art=map_field(ax,ref,ref['elevation']);fig.colorbar(art,cax=ax.inset_axes([1.06,.08,.05,.84]),label='Elevation (m)')
    ax.set_title('Digital Elevation Model');gauges(ax,ref,label_position='bottom_right')
    styles={'full':('tab:blue','-'),'vanilla':('tab:orange','--'),'terrain':('tab:green',':')}
    allvals=np.concatenate([truth.ravel()]+[p.ravel() for p in predictions.values()]);lim=(min(0.,allvals.min()-.1),allvals.max()+.15)
    handles=None
    for j,rep in enumerate(reps):
        x,y=ref['xy'][rep['idx']];ax.scatter(x,y,s=90,c='tab:blue' if j%2==0 else 'tab:orange',marker='o' if j%2==0 else 's',edgecolors='black',zorder=3)
        ax.text(x+1.5,y+2.5,rep['label'],fontsize=12,weight='bold')
        a=fig.add_subplot(gs[j//2,0 if j%2==0 else 2]);a.plot(times,truth[:,j],color='black',lw=2.4,label='SynxFlow')
        for k,(color,ls) in styles.items():a.plot(times,predictions[k][:,j],color=color,ls=ls,lw=2.2,label=LABELS[k])
        metric=f"Peak depth | peak={rep['peak_m']:.2f} m" if j%2==0 else f"Total variation | TV={rep['tv_m']:.2f} m"
        a.set_title(f"({rep['label']}) {metric}");a.set_ylabel('Depth (m)');a.set_ylim(lim);format_time(a,times)
        if j//2==3:a.set_xlabel('Date')
        handles=a.get_legend_handles_labels()
    leg.legend(*handles,loc='center',ncol=2,frameon=False,fontsize=10)
    save(fig,PREFIX+'fig03_diagnostics.png')


def gauge_panels(ref,key,validation=False):
    z=np.load(OUT/f'{key}_traj120.npz');obs=np.load(OUT/'usgs_observations.npz');times=pd.to_datetime(obs['times'][12:])
    sim=obs['sim_depth'][12:]+obs['elevation']+obs['shift_m'];pred=z['point_predictions'][:,:6]+obs['elevation']+obs['shift_m']
    custom=z['usgs_input_predictions']+obs['elevation']+obs['shift_m'];observed=obs['obs_wse'][12:]
    if validation:
        fig=plt.figure(figsize=(12,7.2),constrained_layout=True);gs=GridSpec(3,3,figure=fig,width_ratios=[1.1,1,1])
        vmax=.3*max(ref['snapshot_depth'].max(),z['snapshot_prediction'].max())
        ax=fig.add_subplot(gs[:,0]);art=map_field(ax,ref,ref['snapshot_depth'],cmap='Blues',vmin=0,vmax=vmax);gauges(ax,ref)
        ax.set_title(f"SynxFlow, timestep {int(ref['snapshot_index'])}");fig.colorbar(art,ax=ax,label='Depth (m)',shrink=.7,extend='max')
    else:
        fig=plt.figure(figsize=(10.8,7.2),constrained_layout=True);gs=GridSpec(3,2,figure=fig)
    for i in range(6):
        ax=fig.add_subplot(gs[i//2,i%2+(1 if validation else 0)])
        ax.plot(times,sim[:,i],c='tab:blue',lw=2,label='SynxFlow')
        ax.plot(times,pred[:,i],c='tab:blue',ls='--',lw=2,label='C-STRIDE (sim. input)')
        ax.plot(times,observed[:,i],c='black',lw=2,label='USGS hourly')
        if not validation:ax.plot(times,custom[:,i],c='black',ls='--',lw=2,label='C-STRIDE (USGS input)')
        role=(' | Held out' if i in (3,5) else ' | Input') if key=='four_gauges' else ''
        ax.set_title(f'{i+1}. {NAMES[i]} ({STATIONS[i]}){role}',fontsize=11)
        ax.set_ylabel('Mean-adjusted WSE (m)');format_time(ax,times)
        if i>=4:ax.set_xlabel('Date')
    handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3 if validation else 2,frameon=False)
    name=PREFIX+'fig05_simulated_inputs.png' if validation else (PREFIX+'fig06_observed_inputs.png' if key=='full' else 'fig07_four_gauges.png')
    save(fig,name)


def spatial(ref):
    pred=np.load(OUT/'full_traj120.npz')['snapshot_prediction'];truth=ref['snapshot_depth'];diff=pred-truth
    fig,axes=plt.subplots(1,3,figsize=(9,6),constrained_layout=True);# reduce displayed limits by 70%; retain full data and show saturated ends.
    vmax=.3*max(truth.max(),pred.max());dmax=.3*abs(diff).max()
    for ax,val,title in zip(axes,[truth,pred,diff],['SynxFlow','C-STRIDE (full)','Prediction − reference']):
        art=map_field(ax,ref,val,cmap='RdBu_r' if ax is axes[2] else 'Blues',vmin=-dmax if ax is axes[2] else 0,vmax=dmax if ax is axes[2] else vmax)
        ax.set_title(title,fontsize=10)
        if ax is not axes[0]:
            ax.set_ylabel('')
            ax.tick_params(labelleft=False)
        fig.colorbar(art,ax=ax,label='Depth error (m)' if ax is axes[2] else 'Depth (m)',shrink=.65,extend='both' if ax is axes[2] else 'max')
    save(fig,PREFIX+'fig04_spatial_prediction.png')
    (OUT/'spatial_summary.json').write_text(json.dumps(dict(snapshot_index=int(ref['snapshot_index']),
        selection='maximum total reference depth on indices 12–95',mean_error_m=float(diff.mean()),rmse_m=float(np.sqrt((diff**2).mean())),
        maximum_absolute_error_m=float(abs(diff).max()),evaluation_cells=len(diff)),indent=2))


def tables():
    records=[];common=[];gaugerows=[];obsrows=[];obs=np.load(OUT/'usgs_observations.npz')
    for key in RUNS:
        for tid in ([120] if key=='four_gauges' else EVENTS):
            z=np.load(OUT/f'{key}_traj{tid}.npz');counts=z['counts'];tp,fp,fn=counts
            records.append(dict(model=key,trajectory_id=tid,mean_relative_error=z['relative_error'].mean(),mean_rmse_m=z['rmse_m'].mean(),
                csi=tp/(tp+fp+fn),precision=tp/(tp+fp),recall=tp/(tp+fn),bias=(tp+fp)/(tp+fn)))
            if key.startswith('lag') or key=='full':
                mask=z['timesteps']>=24;common.append(dict(model=key,trajectory_id=tid,mean_relative_error=z['relative_error'][mask].mean(),mean_rmse_m=z['rmse_m'][mask].mean()))
            if tid==120 and key in LABELS:
                for i in range(6):
                    p=z['point_predictions'][:,i];t=obs['sim_depth'][12:,i]
                    gaugerows.append(dict(model=key,station=STATIONS[i],name=NAMES[i],nse=nash_sutcliffe_efficiency(p,t),kge=kling_gupta_efficiency(p,t),
                        peak_depth_error_percent=100*abs(p.max()-t.max())/t.max(),peak_timing_steps=int(np.argmax(p)-np.argmax(t))))
            if tid==120 and key in ['full','four_gauges']:
                for i in range(6):
                    reference=obs['obs_wse'][12:,i];base=obs['elevation'][i]+obs['shift_m'][i]
                    for label,series in [('SynxFlow',obs['sim_depth'][12:,i]+base),('sim_input',z['point_predictions'][:,i]+base),('usgs_input',z['usgs_input_predictions'][:,i]+base)]:
                        obsrows.append(dict(model=key,source=label,station=STATIONS[i],name=NAMES[i],
                            nse=nash_sutcliffe_efficiency(series,reference),peak_timing_hours=int(np.argmax(series)-np.argmax(reference)),
                            offset_added_m=obs['shift_m'][i]))
    pd.DataFrame(records).to_csv(OUT/'reconstruction_summary.csv',index=False)
    pd.DataFrame(common).to_csv(OUT/'history_common_support.csv',index=False)
    pd.DataFrame(gaugerows).to_csv(OUT/'gauge_simulator_metrics.csv',index=False)
    pd.DataFrame(obsrows).to_csv(OUT/'usgs_metrics.csv',index=False)


if __name__=='__main__':
    tables()
    curves([(k,LABELS[k]) for k in ['full','terrain','forcing','vanilla']],PREFIX+'fig02_conditioning.png',12)
    curves([(f'lag{k}' if k!=12 else 'full',f'$K+1={k}$') for k in [1,4,6,12,24]],PREFIX+'fig10_history_length.png',24)
    ref=np.load(OUT/'reference_traj120.npz');diagnostic(ref);gauge_panels(ref,'full',True)
    gauge_panels(ref,'full');gauge_panels(ref,'four_gauges');spatial(ref)
