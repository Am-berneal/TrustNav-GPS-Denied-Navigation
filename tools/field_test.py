"""Real-world field test: score a TrustNav trajectory against known floor checkpoints.

You walk (or carry the device along) a route whose checkpoints you measured on the floor (tiles, tape),
and pause still for ~3 s at each checkpoint. The pauses are found in the IMU itself, the estimated
position during each pause is compared with the measured checkpoint, after one rigid alignment
(rotation + translation, no scale: the same Umeyama step evo uses).

waypoints.json:
  {"name": "Hostel corridor rectangle 4.0 x 3.0 m",
   "points": {"A": [0, 0], "B": [4.0, 0], "C": [4.0, 3.0], "D": [0, 3.0]},
   "route": "ABCDABCDABCDA"}          # order of the pauses, first = the still start
  Optional "times": [[t0, t1], ...] seconds from the start of the recording, one per route letter,
  if you prefer to give the pause times instead of detecting them.

Usage:  python tools/field_test.py <imu data.csv> <estimated_trajectory.txt> <waypoints.json> <out_prefix>
                                   [--label NAME] [--compare other_traj.txt NAME ...]
Writes <out_prefix>.json and <out_prefix>.png; prints a markdown table row per trajectory.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from health import load_imu_any  # noqa: E402


def moving(x, n):
    c = np.cumsum(np.r_[0.0, x]); n = max(1, n)
    out = (c[n:] - c[:-n]) / n
    return np.r_[np.full(n // 2, out[0]), out, np.full(n - 1 - n // 2, out[-1])]


def detect_pauses(t, x, min_len=1.5):
    """Still segments: low gyro and low accel variation for at least min_len seconds."""
    dt = float(np.median(np.diff(t))); n = max(3, int(round(0.5 / dt)))
    w = np.linalg.norm(x[:, 0:3], axis=1)
    an = np.linalg.norm(x[:, 3:6], axis=1)
    wm = np.sqrt(moving(w ** 2, n))
    asd = np.sqrt(np.maximum(moving(an ** 2, n) - moving(an, n) ** 2, 0))
    # thresholds relative to the quietest part of the recording, with sane floors
    wq, aq = np.percentile(wm, 5), np.percentile(asd, 5)
    still = (wm < max(0.12, 4 * wq)) & (asd < max(0.35, 4 * aq))
    segs, i = [], 0
    while i < len(still):
        if still[i]:
            j = i
            while j < len(still) and still[j]:
                j += 1
            if t[j - 1] - t[i] >= min_len:
                segs.append([t[i], t[j - 1]])
            i = j
        else:
            i += 1
    merged = []
    for s in segs:  # join pauses split by a twitch shorter than 0.6 s
        if merged and s[0] - merged[-1][1] < 0.6:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return merged


def load_traj(p):
    a = np.loadtxt(p, comments='#')
    return a[:, 0], a[:, 1:4]


def umeyama_rigid(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    H = (src - ms).T @ (dst - md)
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    return R, md - R @ ms


def score(t_traj, p_traj, pauses, wp):
    est = []
    for a, b in pauses:
        m = (t_traj >= a) & (t_traj <= b)
        est.append(p_traj[m].mean(0) if m.any() else p_traj[np.argmin(np.abs(t_traj - (a + b) / 2))])
    est = np.array(est)
    R, tr = umeyama_rigid(est, wp)
    al = est @ R.T + tr
    err = np.linalg.norm(al[:, :2] - wp[:, :2], axis=1)
    return {'checkpoint_rmse_m': float(np.sqrt(np.mean(err ** 2))), 'checkpoint_mean_m': float(err.mean()),
            'checkpoint_max_m': float(err.max()), 'per_checkpoint_m': [round(float(e), 3) for e in err]}, R, tr, al


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('imu'); ap.add_argument('traj'); ap.add_argument('waypoints'); ap.add_argument('out')
    ap.add_argument('--label', default='TrustNav')
    ap.add_argument('--compare', nargs=2, action='append', default=[], metavar=('TRAJ', 'NAME'))
    a = ap.parse_args()
    W = json.load(open(a.waypoints))
    pts = {k: np.array(v + [0.0] * (3 - len(v)), float) for k, v in W['points'].items()}
    route = list(W['route'])
    wp = np.array([pts[c] for c in route])
    t, x = load_imu_any(a.imu)
    t0 = t[0]
    if 'times' in W:
        pauses = [[t0 + p, t0 + q] for p, q in W['times']]
    else:
        pauses = detect_pauses(t, x)
    print('route %s: %d checkpoints, %d still pauses found' % (''.join(route), len(route), len(pauses)))
    for p in pauses:
        print('   pause %6.1f - %6.1f s' % (p[0] - t0, p[1] - t0))
    if len(pauses) != len(route):
        if len(pauses) > len(route):
            # keep the longest pauses, in time order
            idx = sorted(sorted(range(len(pauses)), key=lambda i: pauses[i][1] - pauses[i][0])[-len(route):])
            pauses = [pauses[i] for i in idx]
            print('   more pauses than checkpoints: kept the %d longest' % len(route))
        else:
            sys.exit('found fewer pauses than checkpoints: add "times" to %s or pause longer (>= 2 s)' % a.waypoints)
    spread = float(np.sqrt(np.mean(np.sum((wp[:, :2] - wp[:, :2].mean(0)) ** 2, 1))))
    res = {'name': W.get('name', ''), 'route': ''.join(route), 'duration_s': float(t[-1] - t0),
           'standstill_reference_m': spread, 'pauses_s': [[round(p - t0, 2), round(q - t0, 2)] for p, q in pauses], 'runs': {}}
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 6))
    loop = np.array([pts[c] for c in route])
    ax.plot(loop[:, 0], loop[:, 1], 'k-', lw=2, alpha=.35, label='measured route')
    for k, v in pts.items():
        ax.plot(v[0], v[1], 'ks', ms=9); ax.annotate(k, (v[0], v[1]), xytext=(6, 6), textcoords='offset points', fontsize=12, weight='bold')
    rows = []
    colors = ['#1f6feb', '#d73a49', '#2da44e', '#8250df']
    for k, (p, name) in enumerate([(a.traj, a.label)] + [tuple(c) for c in a.compare]):
        tt, pp = load_traj(p)
        r, R, tr, al = score(tt, pp, pauses, wp)
        res['runs'][name] = r
        path = pp @ R.T + tr
        ax.plot(path[:, 0], path[:, 1], color=colors[k % 4], lw=1, alpha=.8, label='%s (RMSE %.2f m)' % (name, r['checkpoint_rmse_m']))
        ax.plot(al[:, 0], al[:, 1], 'o', color=colors[k % 4], ms=5)
        rows.append('| %s | %s | %d | %.2f m | %.2f m | %.2f m |' % (name, W.get('name', ''), len(route), r['checkpoint_rmse_m'], r['checkpoint_max_m'], spread))
    span = max(np.ptp(wp[:, 0]), np.ptp(wp[:, 1]), 1.0)
    cx, cy = wp[:, 0].mean(), wp[:, 1].mean()
    ax.set_xlim(cx - span, cx + span); ax.set_ylim(cy - span, cy + span)
    ax.set_aspect('equal'); ax.grid(alpha=.3); ax.legend(fontsize=8, loc='upper right')
    ax.set_title('Field test: %s' % W.get('name', '')); ax.set_xlabel('m'); ax.set_ylabel('m')
    plt.tight_layout(); plt.savefig(a.out + '.png', dpi=110)
    json.dump(res, open(a.out + '.json', 'w'), indent=1)
    print('| run | route | checkpoints | RMSE | max | stand-still reference |')
    print('|---|---|---|---|---|---|')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
