"""Our replica of the organiser degradation described in the Track 3 brief:
progressive IMU noise, bias drift, frame dropouts (2 s), then complete IMU failure.
The official organiser script should replace this once available.

Schedule (fractions of the IMU sequence length T, excluding the final blind window):
  clean          0.00 - 0.20
  noise ramp     0.20 - end   noise x1 -> x(noise_max), linear
  bias drift     0.40 - end   gyro + accel bias random walk growing, plus a slow ramp
  spikes         0.30 - end   rare large outliers on random channels
  dropouts       two 2.0 s gaps at 0.55 and 0.78 (rows removed, or zeroed with --zero-dropouts)
  failure        last 5.0 s: timestamps kept, values invalid (nan by default), per the official FAQ
"""
import argparse, json, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('imu'); ap.add_argument('out_dir')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--noise-max', type=float, default=6.0)
    ap.add_argument('--zero-dropouts', action='store_true', help='keep dropout rows but zero them instead of removing')
    ap.add_argument('--blind', type=float, default=5.0)
    ap.add_argument('--blind-mode', choices=['nan', 'zero', 'garbage', 'remove'], default='nan',
                    help='official FAQ: blind-window timestamps stay in the file, values invalid (default nan)')
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    d = np.loadtxt(a.imu, delimiter=',', comments='#')
    t = d[:, 0].astype(np.int64); x = d[:, 1:7].copy()
    ts = (t - t[0]) * 1e-9
    T_all = ts[-1]; T = T_all - a.blind
    n = len(ts); dt = 0.005
    frac = ts / T
    # 1. progressive noise
    mult = np.where(frac < 0.2, 1.0, 1.0 + (a.noise_max - 1.0) * np.clip((frac - 0.2) / 0.8, 0, 1))
    base = np.array([1.6968e-4, 1.6968e-4, 1.6968e-4, 2e-3, 2e-3, 2e-3]) / np.sqrt(dt)
    extra = rng.normal(0, 1, (n, 6)) * base * np.sqrt(np.maximum(mult ** 2 - 1, 0))[:, None]
    x += extra
    # 2. bias drift (random walk with growing intensity + slow ramp)
    on = np.clip((frac - 0.4) / 0.6, 0, 1)
    rw = rng.normal(0, 1, (n, 6)) * np.array([4e-5, 4e-5, 4e-5, 4e-3, 4e-3, 4e-3]) * np.sqrt(dt) * (1 + 4 * on)[:, None]
    rw[frac < 0.4] = 0
    ramp_rate = rng.choice([-1, 1], 6) * np.array([2e-4, 2e-4, 2e-4, 6e-3, 6e-3, 6e-3])  # per second
    drift = np.cumsum(rw, 0) + np.maximum(ts - 0.4 * T, 0)[:, None] * ramp_rate
    x += drift
    # 3. spikes
    sp = (frac > 0.3) & (rng.random(n) < 0.002)
    ch = rng.integers(0, 6, n)
    amp = np.where(ch < 3, 1.5, 25.0) * rng.choice([-1, 1], n)
    x[sp, ch[sp]] += amp[sp]
    # 4. dropouts and 5. failure
    keep = np.ones(n, bool); zero = np.zeros(n, bool)
    drops = []
    for f0 in (0.55, 0.78):
        s0 = f0 * T; m = (ts >= s0) & (ts < s0 + 2.0)
        drops.append([float(s0), float(s0 + 2.0)])
        if a.zero_dropouts: zero |= m
        else: keep &= ~m
    fail_t = T
    blind = ts >= fail_t
    x[zero] = 0.0
    if a.blind_mode == 'remove': keep &= ~blind
    elif a.blind_mode == 'zero': x[blind] = 0.0
    elif a.blind_mode == 'garbage': x[blind] = rng.normal(0, 50, (int(blind.sum()), 6))
    os.makedirs(os.path.join(a.out_dir, 'mav0/imu0'), exist_ok=True)
    with open(os.path.join(a.out_dir, 'mav0/imu0/data.csv'), 'w') as fh:
        fh.write('#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n')
        for i in np.nonzero(keep)[0]:
            if blind[i] and a.blind_mode == 'nan':
                fh.write('%d,nan,nan,nan,nan,nan,nan\n' % t[i])
            else:
                fh.write('%d,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f\n' % (t[i], *x[i]))
    t0 = t[0] * 1e-9
    ev = {'t0': t0, 'clean_end': 0.2 * T, 'noise_start': 0.2 * T, 'bias_start': 0.4 * T, 'spikes_start': 0.3 * T,
          'dropouts': drops, 'failure': fail_t, 'end': T_all, 'noise_max': a.noise_max,
          'spike_count': int(sp.sum()), 'dropout_mode': 'zeroed' if a.zero_dropouts else 'removed', 'blind_mode': a.blind_mode}
    with open(os.path.join(a.out_dir, 'events.json'), 'w') as fh: json.dump(ev, fh, indent=1)
    print('kept', int(keep.sum()), 'of', n, 'rows;', ev)


if __name__ == '__main__':
    main()
