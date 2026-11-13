"""Score an estimated_trajectory.txt against ground truth, mirroring the Track 3 rubric.

ATE  : RMSE of position after SE(3) Umeyama alignment (same as `evo_ape tum gt est -a`)
RPE  : translation / rotation error over 1 s windows (like `evo_rpe ... --delta 1 --delta_unit s`)
Checkpoints : position error at 6 evenly spaced mission times before IMU failure
Blind 5 s   : error during the 5 s after IMU failure (mean, and at +5 s)
Dropouts    : error inside the 2 s gaps and 1 s after each
Format      : pose count, timestamp coverage, unit quaternions
"""
import json, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def load_tum(path):
    """TUM text (t x y z qx qy qz qw) or EuRoC ground-truth CSV (t[ns], p, qw qx qy qz, ...)."""
    if path.endswith('.csv'):
        a = np.loadtxt(path, delimiter=',', comments='#')
        return a[:, 0] * 1e-9, a[:, 1:4], a[:, [5, 6, 7, 4]]
    a = np.loadtxt(path, comments='#')
    return a[:, 0], a[:, 1:4], a[:, 4:8]


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    C = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, S, Vt = np.linalg.svd(C)
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    return R, mu_d - R @ mu_s


def evaluate(est_path, gt_path, events_path):
    te, pe, qe = load_tum(est_path)
    tg, pg, qg = load_tum(gt_path)
    ev = json.load(open(events_path)); t0 = ev['t0']
    # associate: nearest gt within 3 ms
    idx = np.searchsorted(tg, te).clip(1, len(tg) - 1)
    left = np.abs(te - tg[idx - 1]) < np.abs(te - tg[idx])
    idx = np.where(left, idx - 1, idx)
    ok = np.abs(tg[idx] - te) < 0.003
    te, pe, qe, gi = te[ok], pe[ok], qe[ok], idx[ok]
    pg_, qg_ = pg[gi], qg[gi]
    R, tr = umeyama(pe, pg_)
    pa = pe @ R.T + tr
    err = np.linalg.norm(pa - pg_, axis=1)
    rel = te - t0
    Ra = Rot.from_matrix(R) * Rot.from_quat(qe)
    # Orientation error: heading is unobservable from an IMU, so align orientations with their own
    # best-fit world rotation (not the position-based one, which is unreliable when positions are
    # coarse), and report tilt error separately (needs no alignment).
    Rg_all, Re_all = Rot.from_quat(qg_), Rot.from_quat(qe)
    Rw = (Rg_all * Re_all.inv()).mean()
    rot_err = (Rg_all.inv() * (Rw * Re_all)).magnitude() * 180 / np.pi
    up = np.array([0.0, 0.0, 1.0])
    tilt_err = np.degrees(np.arccos(np.clip(np.sum(Rg_all.inv().apply(up) * Re_all.inv().apply(up), 1), -1, 1)))
    out = {'n_poses': int(len(te)), 'ate_rmse': float(np.sqrt(np.mean(err ** 2))), 'ate_max': float(err.max()),
           'ate_rot_deg': float(np.sqrt(np.mean(rot_err ** 2))),
           'tilt_err_deg': float(np.sqrt(np.mean(tilt_err ** 2)))}
    # RPE over 1 s
    step = int(round(1.0 / np.median(np.diff(te))))
    if len(te) > step + 1:
        a, b = np.arange(0, len(te) - step), np.arange(step, len(te))
        dg = pg_[b] - pg_[a]; de = pa[b] - pa[a]
        Rg = Rot.from_quat(qg_); Re = Ra
        # translation error in the local frame of the gt pose at a
        rel_g = Rg[a].inv().apply(dg); rel_e = Re[a].inv().apply(de)
        out['rpe_trans'] = float(np.sqrt(np.mean(np.sum((rel_g - rel_e) ** 2, 1))))
        dR = (Rg[a].inv() * Rg[b]).inv() * (Re[a].inv() * Re[b])
        out['rpe_rot_deg'] = float(np.sqrt(np.mean((dR.magnitude() * 180 / np.pi) ** 2)))
    # checkpoints: 6 evenly spaced times inside the mission (before failure)
    cps = np.linspace(0.12, 0.98, 6) * ev['failure']
    cp_err = [float(err[np.argmin(np.abs(rel - c))]) for c in cps]
    out['checkpoints'] = [{'t': float(c), 'err': e} for c, e in zip(cps, cp_err)]
    out['checkpoint_mean'] = float(np.mean(cp_err))
    # blind window
    bm = rel >= ev['failure']
    if bm.any():
        out['blind_mean'] = float(err[bm].mean()); out['blind_final'] = float(err[bm][-1])
        out['blind_growth'] = float(err[bm][-1] - err[np.nonzero(bm)[0][0]])
        out['blind_coverage_s'] = float(rel[bm][-1] - ev['failure'])
    # dropouts
    dm = np.zeros(len(rel), bool)
    for a0, a1 in ev['dropouts']: dm |= (rel >= a0) & (rel < a1 + 1.0)
    if dm.any(): out['dropout_mean'] = float(err[dm].mean())
    # format checks
    qn = np.abs(np.linalg.norm(qe, axis=1) - 1)
    out['format'] = {'max_quat_norm_err': float(qn.max()), 'monotonic': bool(np.all(np.diff(te) > 0)),
                     'matched_fraction': float(ok.mean())}
    out['_series'] = {'t': rel, 'err': err, 'pa': pa, 'pg': pg_}
    return out


if __name__ == '__main__':
    r = evaluate(sys.argv[1], sys.argv[2], sys.argv[3]); r.pop('_series')
    print(json.dumps(r, indent=1))
