"""Run the ablation ladder + references on all flights, measure budgets, write dashboard/results.json."""
import sys, json, subprocess, os; sys.path.insert(0,'tools'); sys.path.insert(0,'src')
import numpy as np
from evaluate import evaluate
from fit_params import score
from measure import measure
SEQ=['MH_01','MH_03','MH_05','V1_02','V2_01','V2_02','V2_03']
OFF=['--no-zupt','--vmax','0','--drag-k','0','--no-blind-model']
LADDER=[('naive','Raw strapdown integration',['--no-init','--no-trust','--no-tilt','--no-prior']+OFF),
        ('init','+ quiet-window gravity & gyro-bias init',['--no-trust','--no-tilt','--no-prior']+OFF),
        ('trust','+ per-channel trust & spike rejection',['--no-tilt','--no-prior']+OFF),
        ('tilt','+ trust-weighted tilt correction',['--no-prior']+OFF),
        ('prior','+ learned motion prior (velocity & workspace)',OFF),
        ('drag','+ rotor-drag velocity aiding',['--no-zupt','--vmax','0','--no-blind-model']),
        ('full','+ ZUPT, speed bound, blind-window model (TrustNav-IMU)',[])]
KEYS=['ate_rmse','ate_rot_deg','rpe_trans','rpe_rot_deg','checkpoint_mean','blind_mean','blind_final','dropout_mean']
def ds(a,n=8): return np.round(np.asarray(a)[::n],3).tolist()
out={'ladder':[],'sequences':{},'hold':{},'budgets':{}}
for key,label,flags in LADDER:
    per={}
    for s in SEQ:
        d=f'data/deg/{s}'; o=f'data/runs/{s}_{key}.txt'
        extra=['--trace',f'data/runs/{s}_trace.txt'] if key=='full' else []
        subprocess.run(['python3','src/main.py',f'{d}/mav0/imu0/data.csv',o]+flags+extra,check=True)
        r=evaluate(o,f'{d}/gt.txt',f'{d}/events.json'); S=r.pop('_series'); r['score']=score(r); per[s]=r
        if key=='full' or key=='naive':
            sq=out['sequences'].setdefault(s,{})
            sq['events']=json.load(open(f'{d}/events.json'))
            sq[key]={k:r[k] for k in KEYS+['score','checkpoints','n_poses','format']}
            sq[key]['series']={'t':ds(S['t'],40),'err':ds(S['err'],40),'est':ds(S['pa'],40)}
            if key=='full': sq['gt']=ds(S['pg'],40)
    out['ladder'].append({'key':key,'label':label,**{k:float(np.mean([per[s][k] for s in SEQ])) for k in KEYS+['score']},
                          'per_seq':{s:{k:per[s][k] for k in KEYS+['score']} for s in SEQ}})
    print(key, {k:round(out['ladder'][-1][k],2) for k in ['ate_rmse','checkpoint_mean','blind_mean','dropout_mean','score']})
# hold-position reference
hs=[]
for s in SEQ:
    a=np.loadtxt(f'data/runs/{s}_full.txt'); a[:,1:4]=0; a[:,4:8]=[0,0,0,1]
    np.savetxt('/tmp/hold.txt',a,fmt='%.9f'); r=evaluate('/tmp/hold.txt',f'data/deg/{s}/gt.txt',f'data/deg/{s}/events.json'); r.pop('_series'); r['score']=score(r); hs.append(r)
out['hold']={k:float(np.mean([r[k] for r in hs])) for k in KEYS+['score'] if k in hs[0]}
print('hold',{k:round(v,2) for k,v in out['hold'].items()})
for s in SEQ:
    out['budgets'][s]=measure(f'data/deg/{s}/mav0/imu0/data.csv','/tmp/m.txt')
    print(s,out['budgets'][s])
# trust trace for every sequence (downsampled)
for s in SEQ:
    tr=np.loadtxt(f'data/runs/{s}_trace.txt'); t0=out['sequences'][s]['events']['t0']
    out['sequences'][s]['trace']={'t':np.round(tr[::2,0]-t0,2).tolist(),'trust':np.round(tr[::2,1:7],3).tolist(),
        'bg':np.round(tr[::2,13:16],5).tolist(),'spikes':tr[::2,16].astype(int).tolist()}
out['loo']=json.load(open('data/runs/loo.json'))
os.makedirs('dashboard',exist_ok=True)
json.dump(out,open('dashboard/results.json','w'))
print('size',os.path.getsize('dashboard/results.json'))
