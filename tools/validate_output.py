"""TrustNav "spec is law" output guardrails + reproducibility fingerprints.

Checks estimated_trajectory.txt against the Track 3 format and basic physics BEFORE you submit,
writes SHA-256 fingerprints, and saves everything to validation.json (read by build_console.py)
and to metadata.json under "verification".

Usage (from the project folder):
  python tools/validate_output.py estimated_trajectory.txt real_runs/MH_01_easy_s1/mav0/imu0/data.csv
Options: --blind 5  --vmax 3.0  --src src/main.cpp  --meta metadata.json  --json validation.json
Exit code 0 = all checks passed, 1 = at least one failed.
"""
import argparse
import hashlib
import json
import math
import sys


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('traj'); ap.add_argument('imu')
    ap.add_argument('--blind', type=float, default=5.0)
    ap.add_argument('--vmax', type=float, default=3.0, help='solver speed bound [m/s]')
    ap.add_argument('--src', default='src/main.cpp')
    ap.add_argument('--meta', default='metadata.json')
    ap.add_argument('--json', default='validation.json')
    a = ap.parse_args()

    # read trajectory
    rows, bad_cols, first_bad = [], 0, None
    with open(a.traj) as fh:
        for n, line in enumerate(fh, 1):
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            p = s.split()
            try:
                v = [float(x) for x in p]
            except ValueError:
                v = None
            if v is None or len(p) != 8 or not all(math.isfinite(x) for x in v):
                bad_cols += 1
                first_bad = first_bad or n
                continue
            rows.append(v)
    # read IMU timestamps
    imu_t = []
    with open(a.imu) as fh:
        for line in fh:
            if not line.strip() or line[0] == '#' or line[0].isalpha():
                continue
            t = line.split(',')[0]
            imu_t.append(int(t) * 1e-9 if '.' not in t else float(t))

    t = [r[0] for r in rows]
    checks = []

    def add(rule, name, ok, detail):
        checks.append({'rule': rule, 'name': name, 'pass': bool(ok), 'detail': detail})

    add('FORMAT_TUM', 'Format: 8 numeric columns, space-separated, TUM',
        bad_cols == 0 and len(rows) > 0,
        '%d poses, %d malformed lines%s' % (len(rows), bad_cols, '' if not first_bad else ' (first at line %d)' % first_bad))

    nonmono = sum(1 for i in range(1, len(t)) if t[i] <= t[i - 1])
    add('MONOTONIC_CLOCK', 'Timestamps in seconds, strictly increasing',
        nonmono == 0 and t and t[0] > 1e8,
        'first %.6f s, %d out-of-order rows' % (t[0] if t else 0, nonmono))

    # every IMU frame has a pose (within 0.5 ms), and the blind window is covered
    j, missing = 0, 0
    for ti in imu_t:
        while j < len(t) and t[j] < ti - 5e-4:
            j += 1
        if j >= len(t) or abs(t[j] - ti) > 5e-4:
            missing += 1
    dt = sorted(imu_t[i + 1] - imu_t[i] for i in range(min(len(imu_t) - 1, 2000)))[min(len(imu_t) - 2, 1000)] if len(imu_t) > 2 else 0.005
    blind_cov = (t[-1] - imu_t[-1]) if t and imu_t else 0.0
    add('FRAME_DENSITY_SYNC', 'One pose per IMU frame + %.0f s blind window' % a.blind,
        missing == 0 and blind_cov >= a.blind - 1.5 * dt,
        '%d IMU rows in, %d poses out, %d IMU frames without a pose, blind window %.2f s' % (len(imu_t), len(rows), missing, blind_cov))

    qerr = max((abs(math.sqrt(r[4] ** 2 + r[5] ** 2 + r[6] ** 2 + r[7] ** 2) - 1.0) for r in rows), default=1.0)
    add('SO3_NORM', 'Unit quaternions', qerr < 1e-6, 'max |q|-1 = %.1e' % qerr)

    vmax_seen, jump, jump_lim_at = 0.0, 0.0, 0.0
    for i in range(1, len(rows)):
        d = math.dist(rows[i][1:4], rows[i - 1][1:4])
        h = rows[i][0] - rows[i - 1][0]
        if h > 0:
            vmax_seen = max(vmax_seen, d / h)
        if d > jump:
            jump, jump_lim_at = d, h
    add('DYNAMICS_BOUNDS', 'Speed never above %.1f m/s (physical limit)' % a.vmax,
        vmax_seen <= a.vmax * 1.05, 'max speed %.3f m/s' % vmax_seen)
    jump_lim = a.vmax * 1.05 * max(jump_lim_at, dt)
    add('CONTINUITY_DELTA', 'No position jump between rows beyond what %.1f m/s allows' % a.vmax,
        jump <= jump_lim + 1e-9, 'largest step %.4f m (allowed %.4f m)' % (jump, jump_lim))

    hashes = {}
    for k, pth in (('src/main.cpp', a.src), ('input', a.imu), ('estimated_trajectory.txt', a.traj)):
        try:
            hashes[k] = {'path': pth, 'sha256': sha256(pth)}
        except OSError:
            hashes[k] = {'path': pth, 'sha256': None}

    n_pass = sum(c['pass'] for c in checks)
    out = {'checks': checks, 'passed': n_pass, 'total': len(checks), 'hashes': hashes,
           'imu_rows': len(imu_t), 'poses': len(rows)}
    with open(a.json, 'w') as fh:
        json.dump(out, fh, indent=1)
    try:
        meta = json.load(open(a.meta))
        meta['verification'] = {'checks_passed': '%d/%d' % (n_pass, len(checks)),
                                'sha256': {k: v['sha256'] for k, v in hashes.items()},
                                'reproduce': 'g++ -O2 -std=c++17 -o trustnav src/main.cpp && ./trustnav <input> out.txt && sha256sum out.txt'}
        json.dump(meta, open(a.meta, 'w'), indent=1)
    except (OSError, ValueError):
        pass

    for c in checks:
        print('%-4s %-20s %s' % ('PASS' if c['pass'] else 'FAIL', c['rule'], c['detail']))
    for k, v in hashes.items():
        print('sha256 %-26s %s' % (k, v['sha256']))
    print('%d/%d checks passed -> %s' % (n_pass, len(checks), a.json))
    sys.exit(0 if n_pass == len(checks) else 1)


if __name__ == '__main__':
    main()
