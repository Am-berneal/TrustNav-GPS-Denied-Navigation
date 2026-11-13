"""Stress test: run TrustNav on many jamming scenarios of one real sequence and tabulate.

Usage (from the project folder):  python tools/stress_test.py MH_01_easy [--cpp .\\trustnav.exe]
Writes stress_results/stress_table.md
"""
import json, os, subprocess, sys

SCEN = [
    ('baseline noise x6, seed 1', ['--seed', '1']),
    ('baseline noise x6, seed 2', ['--seed', '2']),
    ('baseline noise x6, seed 3', ['--seed', '3']),
    ('light jamming x3', ['--seed', '1', '--noise-max', '3']),
    ('heavy jamming x10', ['--seed', '1', '--noise-max', '10']),
    ('dropouts zeroed (not removed)', ['--seed', '1', '--zero-dropouts']),
]
KEYS = [('ate_rmse', 'ATE m'), ('checkpoint_mean', 'checkpoints m'), ('blind_mean', 'blind m'),
        ('dropout_mean', 'dropouts m'), ('rpe_trans', 'RPE 1 s m')]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    seq = sys.argv[1]
    cpp = sys.argv[sys.argv.index('--cpp') + 1] if '--cpp' in sys.argv else None
    py = sys.executable
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    imu = os.path.join(seq, 'mav0', 'imu0', 'data.csv')
    gt = os.path.join(seq, 'mav0', 'state_groundtruth_estimate0', 'data.csv')
    lines = ['| scenario | ' + ' | '.join(k[1] for k in KEYS) + ' |', '|---' * (len(KEYS) + 1) + '|']
    for i, (name, opts) in enumerate(SCEN):
        out = os.path.join('stress_results', 'run%d' % i)
        subprocess.run([py, os.path.join(here, 'degrade.py'), imu, out] + opts, check=True, stdout=subprocess.DEVNULL)
        d_imu = os.path.join(out, 'mav0', 'imu0', 'data.csv')
        est = os.path.join(out, 'estimated_trajectory.txt')
        cmd = [cpp, d_imu, est] if cpp else [py, os.path.join(root, 'src', 'main.py'), d_imu, est]
        subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
        r = json.loads(subprocess.run([py, os.path.join(here, 'evaluate.py'), est, gt, os.path.join(out, 'events.json')],
                                      check=True, capture_output=True, text=True).stdout)
        lines.append('| %s | ' % name + ' | '.join('%.2f' % r[k] for k, _ in KEYS) + ' |')
        print(lines[-1], flush=True)
    os.makedirs('stress_results', exist_ok=True)
    with open(os.path.join('stress_results', 'stress_table.md'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('-> stress_results/stress_table.md')


if __name__ == '__main__':
    main()