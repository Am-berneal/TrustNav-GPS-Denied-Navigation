"""Regenerate 200 Hz IMU readings from a real EuRoC ground-truth trajectory.

The flight path (position + orientation) is the real EuRoC motion-capture / laser-tracker
ground truth. Angular rate and specific force are derived from it, then EuRoC's published
ADIS16448 noise model (Kalibr sensor.yaml) is added, plus motor vibration while airborne.

Output (EuRoC ASL layout):
  <out>/mav0/imu0/data.csv                       #timestamp [ns], w_RS_S_x..z, a_RS_S_x..z
  <out>/mav0/state_groundtruth_estimate0/gt.txt  TUM format: t tx ty tz qx qy qz qw
"""
import argparse, os
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot

G = 9.80665
# EuRoC sensor.yaml (ADIS16448) continuous-time noise
GYRO_ND, GYRO_RW = 1.6968e-4, 1.9393e-5
ACC_ND, ACC_RW = 2.0e-3, 3.0e-3


def load_gt(path):
    if path.endswith('.csv'):  # EuRoC ASL: t[ns], p, q(w,x,y,z), ...
        a = np.loadtxt(path, delimiter=',', comments='#')
        t = a[:, 0] * 1e-9; p = a[:, 1:4]; q = a[:, [5, 6, 7, 4]]  # -> x,y,z,w
    else:  # TUM: t, p, qx qy qz qw
        a = np.loadtxt(path, comments='#')
        t = a[:, 0]; p = a[:, 1:4]; q = a[:, 4:8]
    # make quaternion sign continuous
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0: q[i] = -q[i]
    return t, p, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('gt'); ap.add_argument('out'); ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    t, p, q = load_gt(args.gt)
    # fit one smooth model to the real trajectory; derive the reference path AND the IMU from it,
    # so ground truth and IMU are mathematically consistent
    from scipy.interpolate import make_smoothing_spline
    from scipy.spatial.transform import RotationSpline
    tg = np.arange(t[0] + 0.05, t[-1] - 0.05, 0.005)
    tr = t - t[0]; tgr = tg - t[0]
    spl = [make_smoothing_spline(tr, p[:, i], lam=2e-4) for i in range(3)]
    p = np.stack([s_(tgr) for s_ in spl], 1)
    vel = np.stack([s_.derivative(1)(tgr) for s_ in spl], 1)
    acc = np.stack([s_.derivative(2)(tgr) for s_ in spl], 1)
    key = np.arange(0, len(t), 10)  # 20 Hz key orientations, smooth spline between them
    rs = RotationSpline(tr[key], Rot.from_quat(q[key]))
    R = rs(tgr)
    gyro = rs(tgr, 1)  # SciPy returns body-frame angular rate (checked against finite differences)
    t = tg; dt = 0.005
    # specific force in body frame: f = R^T (a - g),  g = (0,0,-G)
    f = R.inv().apply(acc - np.array([0, 0, -G]))
    n = len(t)
    # sensor errors: constant turn-on bias + random walk + white noise (EuRoC sensor.yaml)
    bg0 = rng.normal(0, 0.02, 3); ba0 = rng.normal(0, 0.08, 3)
    bg = bg0 + np.cumsum(rng.normal(0, GYRO_RW * np.sqrt(dt), (n, 3)), 0)
    ba = ba0 + np.cumsum(rng.normal(0, ACC_RW * np.sqrt(dt), (n, 3)), 0)
    wn = rng.normal(0, GYRO_ND / np.sqrt(dt), (n, 3))
    an = rng.normal(0, ACC_ND / np.sqrt(dt), (n, 3))
    # motor vibration while airborne (speed > 0.15 m/s or height change): narrow-band, ~0.25 m/s^2
    air = np.clip((np.linalg.norm(vel, axis=1) - 0.05) / 0.2, 0, 1)
    air = np.convolve(air, np.ones(200) / 200, mode='same')
    ph = rng.uniform(0, 2 * np.pi, 6)
    vib = np.stack([0.25 * np.sin(2 * np.pi * (61 + 3 * i) * t + ph[i]) + 0.1 * rng.normal(0, 1, n) for i in range(3)], 1)
    gvib = np.stack([0.01 * np.sin(2 * np.pi * (67 + 2 * i) * t + ph[3 + i]) for i in range(3)], 1)
    gm = gyro + bg + wn + gvib * air[:, None]
    am = f + ba + an + vib * air[:, None]
    os.makedirs(os.path.join(args.out, 'mav0/imu0'), exist_ok=True)
    os.makedirs(os.path.join(args.out, 'mav0/state_groundtruth_estimate0'), exist_ok=True)
    tns = np.round(t * 1e9).astype(np.int64)
    with open(os.path.join(args.out, 'mav0/imu0/data.csv'), 'w') as fh:
        fh.write('#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n')
        for i in range(n):
            fh.write('%d,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f\n' % (tns[i], *gm[i], *am[i]))
    qq = R.as_quat()
    with open(os.path.join(args.out, 'mav0/state_groundtruth_estimate0/gt.txt'), 'w') as fh:
        fh.write('# timestamp tx ty tz qx qy qz qw\n')
        for i in range(n):
            fh.write('%.9f %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n' % (tns[i] * 1e-9, *p[i], *qq[i]))
    np.savez_compressed(os.path.join(args.out, 'truth_extra.npz'), bg=bg, ba=ba, vel=vel)
    print(args.out, n, 'samples', '%.1f s' % (t[-1] - t[0]))


if __name__ == '__main__':
    main()
