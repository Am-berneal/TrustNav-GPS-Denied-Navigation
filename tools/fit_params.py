"""Fit the motion-prior / filter parameters on training flights (leave-one-out capable)."""
import sys, os, itertools, json, random
sys.path.insert(0, 'src'); sys.path.insert(0, 'tools')
import numpy as np
import main as M
from evaluate import evaluate
SEQ = ['MH_01','MH_03','MH_05','V1_02','V2_01','V2_02','V2_03']
OUT = '/tmp/fit_%d' % os.getpid(); os.makedirs(OUT, exist_ok=True)
def score(r):  # rubric-weighted trajectory closeness: waypoints 30, blind 20, dropout 10
    return (30*r['checkpoint_mean'] + 20*r['blind_mean'] + 10*r['dropout_mean'])/60
def run(params, seqs, extra=()):
    res = {}
    for s in seqs:
        d = f'data/deg/{s}'; o = f'{OUT}/{s}.txt'
        argv = [f'{d}/mav0/imu0/data.csv', o] + list(extra)
        for k, v in params.items(): argv += ['--' + k.replace('_','-'), str(v)]
        M.run(M.parse(argv))
        r = evaluate(o, f'{d}/gt.txt', f'{d}/events.json'); r.pop('_series'); res[s] = r
    return res
GRID = dict(tau_v=[1,2,4,8], tau_vz=[0.5,1,2], tau_p=[2,4,8,16], tau_c=[0,20,60], kp=[0.02,0.1], tau_b=[0.5,1,2])
def search(seqs, n=120, seed=0):
    rnd = random.Random(seed); keys = list(GRID); best = None
    combos = list(itertools.product(*GRID.values())); rnd.shuffle(combos)
    for c in combos[:n]:
        p = dict(zip(keys, c))
        res = run(p, seqs); sc = np.mean([score(r) for r in res.values()])
        if best is None or sc < best[0]: best = (sc, p)
    return best
if __name__ == '__main__':
    best = search(SEQ, n=int(sys.argv[1]) if len(sys.argv) > 1 else 120)
    print(best)
