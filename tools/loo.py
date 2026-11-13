import sys, json; sys.path.insert(0,'tools'); sys.path.insert(0,'src')
from fit_params import *
top=json.load(open('/tmp/top.json'))
per=[]
for sc,ate,p in top:
    res=run(p,SEQ); per.append({s:score(res[s]) for s in SEQ}|{'_ate':{s:res[s]['ate_rmse'] for s in SEQ}})
out={}
for s in SEQ:
    i=int(np.argmin([np.mean([d[o] for o in SEQ if o!=s]) for d in per]))
    out[s]={'config':i,'heldout_score':per[i][s],'heldout_ate':per[i]['_ate'][s]}
    print(s,out[s])
print('LOO mean score %.2f ate %.2f'%(np.mean([v['heldout_score'] for v in out.values()]),np.mean([v['heldout_ate'] for v in out.values()])))
json.dump(out,open('data/runs/loo.json','w'),indent=1)
