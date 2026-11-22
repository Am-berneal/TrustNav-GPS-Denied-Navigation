"""Run the organisers' degrade_imu.py at several levels and in both column orders, then solve, validate and score.
Usage: python tools/official_check.py MH_01_easy [path/to/degrade_imu.py]   -> real_runs/official_table.md"""
import json, os, subprocess, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import evaluate
seq = sys.argv[1]; script = sys.argv[2] if len(sys.argv) > 2 else 'tools/official/degrade_imu.py'
gt = os.path.join(seq, 'mav0/state_groundtruth_estimate0/data.csv')
raw = os.path.join(seq, 'mav0/imu0/data.csv')
work = 'real_runs/official_check'; os.makedirs(work, exist_ok=True)
d = pd.read_csv(raw); c = list(d.columns)
accfirst = os.path.join(work, 'clean_accel_first.csv'); d[[c[0]] + c[4:7] + c[1:4]].to_csv(accfirst, index=False)
t0 = d[c[0]].iloc[0] * 1e-9; T = (d[c[0]].iloc[-1] - d[c[0]].iloc[0]) * 1e-9
ev = os.path.join(work, 'events.json'); json.dump({'t0': t0, 'failure': T - 5.0, 'end': T, 'dropouts': []}, open(ev, 'w'))
rows = []
for name, src, extra in [('documented order (t, accel, gyro), level 1.0', accfirst, []),
                         ('raw EuRoC order (t, gyro, accel), level 1.0', raw, []),
                         ('documented order, level 0.5', accfirst, ['--max-level', '0.5']),
                         ('documented order, level 2.0', accfirst, ['--max-level', '2.0']),
                         ('documented order, level 1.0, gamma 1.0', accfirst, ['--gamma', '1.0'])]:
    k = len(rows); deg = os.path.join(work, 'deg%d.csv' % k); est = os.path.join(work, 'est%d.txt' % k)
    subprocess.run([sys.executable, script, src, deg] + extra, check=True, capture_output=True)
    subprocess.run([sys.executable, 'src/main.py', deg, est], check=True)
    v = subprocess.run([sys.executable, 'tools/validate_output.py', est, deg, '--meta', os.path.join(work, 'm.json'),
                        '--json', os.path.join(work, 'v%d.json' % k)], capture_output=True, text=True).stdout
    nin = sum(1 for l in open(deg) if l[0].isdigit()); nout = sum(1 for l in open(est) if l[0].isdigit())
    r = evaluate(est, gt, ev)
    rows.append('| %s | %d / %d | %s | %.2f | %.2f | %.2f | %.2f |' % (name, nin, nout, v.strip().splitlines()[-1].split(' ->')[0],
                r['ate_rmse'], r['checkpoint_mean'], r['blind_mean'], r['rpe_trans']))
    print(rows[-1], flush=True)
out = '| official degrade_imu.py run | rows in / poses out | guardrails | ATE m | checkpoints m | blind m | RPE 1 s m |\n|---|---|---|---|---|---|---|\n' + '\n'.join(rows) + '\n'
open('real_runs/official_table.md', 'w').write(out)
