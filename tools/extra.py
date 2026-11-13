"""Extra experiments for the dashboard: init per flight, blind-model sweep, stress scenarios, phase errors."""
import sys, os, json, subprocess, io, contextlib
sys.path.insert(0,'tools'); sys.path.insert(0,'src')
import numpy as np, main as M
from evaluate import evaluate
from fit_params import score
SEQ=['MH_01','MH_03','MH_05','V1_02','V2_01','V2_02','V2_03']
X={}
# 1. start-up calibration per flight
X['init']={}
for s in SEQ:
    cfg=M.parse([f'data/deg/{s}/mav0/imu0/data.csv','/tmp/x.txt','-v'])
    err=io.StringIO()
    with contextlib.redirect_stderr(err): M.run(cfg)
    line=err.getvalue().strip().split('init: ',1)[1]; X['init'][s]=eval(line)
    X['init'][s]['spikes']=int(err.getvalue().split('spikes rejected: ')[1].split(',')[0])
    print(s,X['init'][s])
# 2. blind-window model sweep
def runset(extra, deg='data/deg', tag='x'):
    res={}
    for s in SEQ:
        o=f'/tmp/ex_{tag}_{s}.txt'; M.run(M.parse([f'{deg}/{s}/mav0/imu0/data.csv',o]+extra))
        r=evaluate(o,f'data/deg/{s}/gt.txt',f'{deg}/{s}/events.json'); r.pop('_series'); r['score']=score(r); res[s]=r
    return res
X['blind_sweep']=[]
for tb in [0.25,0.5,1,2,4,1000]:
    r=runset(['--tau-b',str(tb)],tag='b')
    X['blind_sweep'].append({'tau_b':tb,'blind_mean':float(np.mean([v['blind_mean'] for v in r.values()])),'blind_final':float(np.mean([v['blind_final'] for v in r.values()]))})
    print('tau_b',X['blind_sweep'][-1])
r=runset(['--no-blind-model'],tag='b'); X['blind_sweep_nomodel']={'blind_mean':float(np.mean([v['blind_mean'] for v in r.values()])),'blind_final':float(np.mean([v['blind_final'] for v in r.values()]))}
print('no model',X['blind_sweep_nomodel'])
# 3. velocity-prior sweep
X['tau_v_sweep']=[]
for tv in [0.5,1,2,4,8]:
    r=runset(['--tau-v',str(tv)],tag='v'); X['tau_v_sweep'].append({'tau_v':tv,'ate':float(np.mean([v['ate_rmse'] for v in r.values()])),'score':float(np.mean([v['score'] for v in r.values()]))}); print(X['tau_v_sweep'][-1])
X['tau_p_sweep']=[]
for tp in [1.5,3,6,12,0]:
    r=runset(['--tau-p',str(tp)],tag='p'); X['tau_p_sweep'].append({'tau_p':tp,'ate':float(np.mean([v['ate_rmse'] for v in r.values()])),'score':float(np.mean([v['score'] for v in r.values()]))}); print(X['tau_p_sweep'][-1])
# 4. stress scenarios: regenerate degradation
SC={'noise_x3':['--noise-max','3'],'noise_x10':['--noise-max','10'],'zeroed_dropouts':['--zero-dropouts'],'new_seed':['--seed','7']}
X['stress']={}
for name,args in SC.items():
    deg=f'/tmp/stress_{name}'
    for s in SEQ:
        os.makedirs(f'{deg}/{s}',exist_ok=True)
        subprocess.run(['python3','tools/degrade.py',f'data/seq/{s}/mav0/imu0/data.csv',f'{deg}/{s}']+args,check=True,capture_output=True)
    rows={}
    for key,fl in [('naive',['--no-init','--no-trust','--no-tilt','--no-prior','--no-zupt','--vmax','0','--drag-k','0','--no-blind-model']),('full',[])]:
        r=runset(fl,deg=deg,tag=name+key)
        rows[key]={k:float(np.mean([v[k] for v in r.values()])) for k in ['ate_rmse','checkpoint_mean','blind_mean','blind_final','dropout_mean','score']}
    # hold
    hs=[]
    for s in SEQ:
        a=np.loadtxt(f'/tmp/ex_{name}full_{s}.txt'); a[:,1:4]=0; a[:,4:8]=[0,0,0,1]; np.savetxt('/tmp/h.txt',a,fmt='%.9f')
        r=evaluate('/tmp/h.txt',f'data/deg/{s}/gt.txt',f'{deg}/{s}/events.json'); r.pop('_series'); r['score']=score(r); hs.append(r)
    rows['hold']={k:float(np.mean([h[k] for h in hs])) for k in rows['full']}
    X['stress'][name]=rows; print(name,{k:{kk:round(vv,2) for kk,vv in v.items()} for k,v in rows.items()})
json.dump(X,open('dashboard/extra.json','w'),indent=1)
