#!/usr/bin/env python3
"""TrustNav-IMU: 6-DoF trajectory from a degraded IMU stream.

Input : EuRoC-format IMU CSV (imu0/data.csv): t[ns], gyro xyz [rad/s], accel xyz [m/s^2]
Output: TUM poses "timestamp tx ty tz qx qy qz qw", one per IMU frame, gaps filled at the
        nominal rate, plus a 5 s dead-reckoned extension after the IMU stops.

Python standard library only; single pass; constant memory; single thread; offline.

Usage: python3 main.py imu0/data.csv estimated_trajectory.txt [options]
"""
import argparse
import math
import sys

G0 = 9.80665


# ---------------------------------------------------------------- quaternion helpers (w, x, y, z)
def qmul(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def qexp(wx, wy, wz, dt):
    th = math.sqrt(wx * wx + wy * wy + wz * wz) * dt
    if th < 1e-12:
        return (1.0, 0.5 * wx * dt, 0.5 * wy * dt, 0.5 * wz * dt)
    s = math.sin(0.5 * th) / (th / dt)
    return (math.cos(0.5 * th), wx * s, wy * s, wz * s)


def qnorm(q):
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def rot(q, x, y, z):
    """Rotate body vector into world: R(q) v."""
    w, a, b, c = q
    tx = 2 * (b * z - c * y); ty = 2 * (c * x - a * z); tz = 2 * (a * y - b * x)
    return (x + w * tx + b * tz - c * ty, y + w * ty + c * tx - a * tz, z + w * tz + a * ty - b * tx)


def rot_inv(q, x, y, z):
    return rot((q[0], -q[1], -q[2], -q[3]), x, y, z)


def q_from_two(u):
    """Quaternion that rotates unit body vector u onto world up (0,0,1)."""
    ux, uy, uz = u
    d = uz
    if d < -0.999999:
        return (0.0, 1.0, 0.0, 0.0)
    cx, cy, cz = uy * 1.0 - uz * 0.0, uz * 0.0 - ux * 1.0, 0.0  # u x z
    return qnorm((1.0 + d, cx, cy, cz))


# ---------------------------------------------------------------- streaming reader
def read_rows(path):
    with open(path, 'r') as fh:
        for line in fh:
            if not line or line[0] == '#' or line[0].isalpha():
                continue
            p = line.split(',')
            if len(p) < 7:
                continue
            try:
                t = int(p[0]) * 1e-9 if '.' not in p[0] else float(p[0])
                v = [float(s) for s in p[1:7]]
            except ValueError:
                continue
            ok = all(x == x and abs(x) < 1e4 for x in v) and any(x != 0.0 for x in v)
            yield t, v, ok


# ---------------------------------------------------------------- estimator
class TrustNav:
    def __init__(self, cfg):
        self.c = cfg
        self.q = (1.0, 0.0, 0.0, 0.0)
        self.v = [0.0, 0.0, 0.0]
        self.p = [0.0, 0.0, 0.0]
        self.bg = [0.0, 0.0, 0.0]
        self.g = G0
        self.center = [0.0, 0.0, 0.0]
        # trust: per channel low-pass, noise level, baseline noise
        self.lp = [0.0] * 6
        self.nl = [0.0] * 6
        self.base = [1e-3] * 6
        self.trust = [1.0] * 6
        self.spikes = 0
        self.w_last = [0.0, 0.0, 0.0]
        self.t = None
        self.dt_nom = 0.005
        self.n_b = (1.0, 0.0, 0.0)      # thrust axis in body frame (from the take-off attitude)
        self.f_lp = [0.0, 0.0, 0.0]     # 0.1 s low-pass specific force
        self.f_long = [0.0, 0.0, 0.0]   # slow mean: tracks accelerometer bias drift
        self.gap_t = 0.0
        self.za = G0; self.zv = 1.0; self.zw = 1.0; self.still = False
        self.drag_w = 0.0               # how much the drag observation is currently trusted

    # -- initialisation: find the quietest 1 s window in the first seconds; gravity direction,
    #    gyro bias and noise baseline come from it; the attitude is carried back to the first frame
    def initialise(self, rows):
        good = [r for r in rows if r[2]] or rows
        n = len(good)
        dts = sorted(good[i + 1][0] - good[i][0] for i in range(min(n - 1, 400)))
        if dts:
            self.dt_nom = max(1e-4, dts[len(dts) // 2])
        W = max(20, int(round(self.c.init_s / self.dt_nom)))
        best = None
        for s0 in range(0, max(1, n - W + 1), max(1, W // 4)):
            seg = good[s0:s0 + W]
            m = len(seg)
            mw = [sum(r[1][i] for r in seg) / m for i in range(3)]
            gs = math.sqrt(sum(sum((r[1][i] - mw[i]) ** 2 for r in seg) / m for i in range(3)))
            an = [math.sqrt(r[1][3] ** 2 + r[1][4] ** 2 + r[1][5] ** 2) for r in seg]
            am = sum(an) / m
            as_ = math.sqrt(sum((x - am) ** 2 for x in an) / m)
            score = gs / 0.02 + as_ / 0.1
            if best is None or score < best[0]:
                best = (score, s0, gs, as_)
        _, s0, gs, as_ = best
        seg = good[s0:s0 + W]
        m = len(seg)
        mw = [sum(r[1][i] for r in seg) / m for i in range(3)]
        ma = [sum(r[1][3 + i] for r in seg) / m for i in range(3)]
        anorm = math.sqrt(sum(x * x for x in ma))
        self.static = gs < 0.05 and as_ < 0.15
        u = (ma[0] / anorm, ma[1] / anorm, ma[2] / anorm)
        if self.c.init:
            if self.static:
                self.bg = mw[:]
                self.g = anorm
            q = q_from_two(u)
            # carry the attitude from the quiet window back to the first frame (reverse gyro integration)
            for i in range(s0 + m // 2, 0, -1):
                dt = good[i][0] - good[i - 1][0]
                w = good[i - 1][1]
                q = qnorm(qmul(q, qexp(-(w[0] - self.bg[0]), -(w[1] - self.bg[1]), -(w[2] - self.bg[2]), dt)))
            self.q = q
        self.n_b = u
        self.f_lp = list(good[0][1][3:6])
        self.f_long = ma[:]
        for i in range(6):
            vals = [r[1][i] for r in seg]
            mm = sum(vals) / m
            self.lp[i] = good[0][1][i]
            d = sorted(abs(x - mm) for x in vals)
            self.base[i] = max(d[len(d) // 2] * 1.4826, 1e-5)
            self.nl[i] = self.base[i]
        self.info = {'static': self.static, 'quiet_window_s': round(good[s0][0] - good[0][0], 2),
                     'g': round(self.g, 4), 'bg': [round(x, 5) for x in self.bg], 'dt': self.dt_nom}

    # -- trust scoring and spike rejection, per channel
    def clean(self, v):
        c = self.c
        out = v[:]
        for i in range(6):
            d = v[i] - self.lp[i]
            ad = abs(d)
            lim = c.spike_k * max(self.nl[i], self.base[i])
            if c.trust and ad > lim and ad > (0.6 if i < 3 else 6.0):
                out[i] = self.lp[i]
                self.spikes += 1
                ad = lim
            self.nl[i] += 0.01 * (ad - self.nl[i])
            self.lp[i] += 0.3 * (out[i] - self.lp[i])
            r = self.base[i] / max(self.nl[i], 1e-9)
            self.trust[i] = r if r < 1.0 else 1.0
        return out

    def step(self, dt, v):
        c = self.c
        wx, wy, wz = v[0] - self.bg[0], v[1] - self.bg[1], v[2] - self.bg[2]
        ax, ay, az = v[3], v[4], v[5]
        # tilt correction: over time, the mean specific force of bounded flight points "up"
        if c.tilt:
            an = math.sqrt(ax * ax + ay * ay + az * az)
            if abs(an - self.g) < c.tilt_gate:
                ux, uy, uz = rot_inv(self.q, 0.0, 0.0, 1.0)
                fx, fy, fz = ax / an, ay / an, az / an
                ex, ey, ez = fy * uz - fz * uy, fz * ux - fx * uz, fx * uy - fy * ux
                tg = min(self.trust[3], self.trust[4], self.trust[5]) if c.trust else 1.0
                k = c.kp * tg
                if self.gap_t > 0:  # just after a dropout: re-level quickly
                    self.gap_t -= dt
                    k = max(k, c.kp_gap)
                wx += k * ex; wy += k * ey; wz += k * ez
                ki = c.ki * tg * dt
                self.bg[0] -= ki * ex; self.bg[1] -= ki * ey; self.bg[2] -= ki * ez
        q1 = qnorm(qmul(self.q, qexp(wx, wy, wz, dt)))
        fa = rot(self.q, ax, ay, az); fb = rot(q1, ax, ay, az)
        axw = 0.5 * (fa[0] + fb[0]); ayw = 0.5 * (fa[1] + fb[1]); azw = 0.5 * (fa[2] + fb[2]) - self.g
        self.q = q1
        vx, vy, vz = self.v
        nvx, nvy, nvz = vx + axw * dt, vy + ayw * dt, vz + azw * dt
        if c.prior:
            # learned motion prior: velocity mean-reverts (fitted time constant), position stays near its working area
            dec = math.exp(-dt / c.tau_v)
            nvx *= dec; nvy *= dec; nvz *= math.exp(-dt / c.tau_vz)
        px, py, pz = self.p
        px += 0.5 * (vx + nvx) * dt; py += 0.5 * (vy + nvy) * dt; pz += 0.5 * (vz + nvz) * dt
        if c.prior and c.tau_p > 0:
            cx, cy, cz = self.center
            if c.tau_c > 0:  # slowly re-centre the workspace
                a = dt / c.tau_c
                cx += a * (px - cx); cy += a * (py - cy); cz += a * (pz - cz)
                self.center = [cx, cy, cz]
            k = dt / c.tau_p
            px -= k * (px - cx); py -= k * (py - cy); pz -= k * (pz - cz) * c.z_spring
        if c.drag_k > 0:
            nvx, nvy, nvz = self.drag_update(dt, ax, ay, az, nvx, nvy, nvz)
        # zero-velocity detector (on the ground / held still): accel magnitude steady, no rotation
        an = math.sqrt(ax * ax + ay * ay + az * az)
        e = dt / 0.3
        self.za += e * (an - self.za)
        self.zv += e * ((an - self.za) ** 2 - self.zv)
        self.zw += e * (math.sqrt(wx * wx + wy * wy + wz * wz) - self.zw)
        self.still = c.zupt and self.zv < c.zupt_a ** 2 and self.zw < c.zupt_w and abs(self.za - self.g) < 0.3
        if self.still:
            nvx = nvy = nvz = 0.0
            px, py, pz = self.p
        if c.vmax > 0:
            sp = math.sqrt(nvx * nvx + nvy * nvy + nvz * nvz)
            if sp > c.vmax:
                s_ = c.vmax / sp; nvx *= s_; nvy *= s_; nvz *= s_
        # physical guardrail: the position may not move faster than the speed bound
        if c.vmax > 0:
            ox, oy, oz = self.p
            dx, dy, dz = px - ox, py - oy, pz - oz
            dn = math.sqrt(dx * dx + dy * dy + dz * dz); lim = c.vmax * dt
            if dn > lim:
                s_ = lim / dn; px, py, pz = ox + dx * s_, oy + dy * s_, oz + dz * s_
        self.v = [nvx, nvy, nvz]; self.p = [px, py, pz]
        self.w_last = [0.98 * self.w_last[i] + 0.02 * (wx, wy, wz)[i] for i in range(3)]

    def drag_update(self, dt, ax, ay, az, vx, vy, vz):
        """Rotor-drag velocity observation (multirotor physics): the specific force perpendicular
        to the thrust axis is proportional to body velocity, f_perp = -k v_perp."""
        c = self.c
        a = dt / c.tau_f
        f = (ax, ay, az)
        for i in range(3):
            self.f_lp[i] += a * (f[i] - self.f_lp[i])
            self.f_long[i] += dt / c.tau_bias * (f[i] - self.f_long[i])
        d = [self.f_lp[i] - self.f_long[i] for i in range(3)]
        n = self.n_b
        dn = d[0] * n[0] + d[1] * n[1] + d[2] * n[2]
        fp = [d[i] - dn * n[i] for i in range(3)]
        vb = rot_inv(self.q, vx, vy, vz)
        vn = vb[0] * n[0] + vb[1] * n[1] + vb[2] * n[2]
        vp = [vb[i] - vn * n[i] for i in range(3)]
        w = min(self.trust[3], self.trust[4], self.trust[5]) if c.trust else 1.0
        self.drag_w = w
        b = min(1.0, c.drag_beta * w * dt)
        vp = [vp[i] + b * (-fp[i] / c.drag_k - vp[i]) for i in range(3)]
        vb = [vp[i] + vn * n[i] for i in range(3)]
        return rot(self.q, vb[0], vb[1], vb[2])

    def coast(self, dt):
        """No IMU data (dropout or failure): dead-reckon with the motion model.
        A hovering multirotor keeps its tilt near level, so only the heading keeps turning
        (decaying yaw rate); velocity decays with the fitted time constant."""
        c = self.c
        self.gap_t = c.relevel_s
        if c.blind_model:
            dec = math.exp(-dt / c.tau_b)
            self.v = [self.v[0] * dec, self.v[1] * dec, self.v[2] * math.exp(-dt / c.tau_vz)]
            wd = math.exp(-dt / c.tau_w)
            self.w_last = [x * wd for x in self.w_last]
            u = rot_inv(self.q, 0.0, 0.0, 1.0)  # world up in body
            wz = self.w_last[0] * u[0] + self.w_last[1] * u[1] + self.w_last[2] * u[2]
            self.q = qnorm(qmul(self.q, qexp(wz * u[0], wz * u[1], wz * u[2], dt)))
        for i in range(3):
            self.p[i] += self.v[i] * dt

    def pose_line(self, t):
        q = self.q
        return '%.9f %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n' % (t, self.p[0], self.p[1], self.p[2], q[1], q[2], q[3], q[0])


def run(cfg):
    est = TrustNav(cfg)
    out = open(cfg.output, 'w')
    out.write('# timestamp tx ty tz qx qy qz qw\n')
    trace = open(cfg.trace, 'w') if cfg.trace else None
    rows = read_rows(cfg.input)
    buf = []
    for r in rows:  # first second (bounded buffer) for initialisation
        buf.append(r)
        if r[0] - buf[0][0] >= cfg.init_search_s or len(buf) >= 4000:
            break
    if not buf:
        out.close(); return
    est.initialise(buf)
    n_written = 0
    last_t = buf[0][0]
    out.write(est.pose_line(last_t)); n_written += 1
    next_trace = last_t

    def handle(r):
        nonlocal last_t, n_written, next_trace
        t, v, ok = r
        dt = t - last_t
        if dt <= 0:
            return
        if cfg.fill and dt > 1.6 * est.dt_nom:
            k = int(round(dt / est.dt_nom)) - 1
            for j in range(k):  # fill the gap at the nominal rate
                tt = last_t + est.dt_nom
                est.coast(est.dt_nom)
                out.write(est.pose_line(tt)); n_written += 1
                last_t = tt
            dt = t - last_t
            if dt <= 0:
                return
        if ok:
            est.step(dt, est.clean(v))
        else:
            est.coast(dt)
        out.write(est.pose_line(t)); n_written += 1
        last_t = t
        if trace and t >= next_trace:
            next_trace = t + 0.05
            trace.write('%.3f %s %s %s %d %d\n' % (t, ' '.join('%.3f' % x for x in est.trust),
                                                  ' '.join('%.5f' % x for x in est.nl), ' '.join('%.5f' % x for x in est.bg),
                                                  est.spikes, 1 if ok else 0))

    for r in buf[1:]:
        handle(r)
    buf = None
    for r in rows:
        handle(r)
    # IMU has stopped: dead-reckon the blind window at the nominal rate
    nb = int(round(cfg.blind / est.dt_nom))
    for j in range(nb):
        last_t += est.dt_nom
        est.coast(est.dt_nom)
        out.write(est.pose_line(last_t)); n_written += 1
    out.close()
    if trace:
        trace.close()
    if cfg.verbose:
        sys.stderr.write('poses written: %d, spikes rejected: %d, init: %s\n' % (n_written, est.spikes, est.info))


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input'); ap.add_argument('output')
    ap.add_argument('--blind', type=float, default=5.0, help='seconds to dead-reckon after the last IMU sample')
    ap.add_argument('--no-fill', dest='fill', action='store_false', help='do not fill timestamp gaps')
    ap.add_argument('--init-s', type=float, default=1.0, help='length of the quiet window')
    ap.add_argument('--init-search-s', type=float, default=10.0, help='search the first N s for the quiet window')
    # components (for ablation)
    ap.add_argument('--no-init', dest='init', action='store_false')
    ap.add_argument('--no-trust', dest='trust', action='store_false')
    ap.add_argument('--no-tilt', dest='tilt', action='store_false')
    ap.add_argument('--no-prior', dest='prior', action='store_false')
    ap.add_argument('--no-blind-model', dest='blind_model', action='store_false')
    # parameters (fitted on training flights, see tools/fit_params.py)
    ap.add_argument('--spike-k', type=float, default=8.0)
    ap.add_argument('--kp', type=float, default=1)
    ap.add_argument('--ki', type=float, default=0.005)
    ap.add_argument('--tilt-gate', type=float, default=3.0)
    ap.add_argument('--tau-v', type=float, default=1)
    ap.add_argument('--tau-vz', type=float, default=0.5)
    ap.add_argument('--tau-p', type=float, default=3)
    ap.add_argument('--tau-c', type=float, default=0)
    ap.add_argument('--z-spring', type=float, default=1.0)
    ap.add_argument('--tau-b', type=float, default=0.5)
    ap.add_argument('--tau-w', type=float, default=3)
    ap.add_argument('--drag-k', type=float, default=0.17, help='rotor-drag coefficient [1/s]; 0 disables')
    ap.add_argument('--drag-beta', type=float, default=1, help='drag observation gain [1/s]')
    ap.add_argument('--tau-f', type=float, default=0.3)
    ap.add_argument('--tau-bias', type=float, default=5)
    ap.add_argument('--no-zupt', dest='zupt', action='store_false')
    ap.add_argument('--zupt-a', type=float, default=0.15)
    ap.add_argument('--zupt-w', type=float, default=0.05)
    ap.add_argument('--vmax', type=float, default=3.0, help='speed bound learned from training flights [m/s]; 0 disables')
    ap.add_argument('--kp-gap', type=float, default=1.0, help='tilt gain right after a dropout')
    ap.add_argument('--relevel-s', type=float, default=3.0)
    ap.add_argument('--trace', default=None, help='optional diagnostics file for the dashboard')
    ap.add_argument('-v', '--verbose', action='store_true')
    return ap.parse_args(argv)


if __name__ == '__main__':
    run(parse())
