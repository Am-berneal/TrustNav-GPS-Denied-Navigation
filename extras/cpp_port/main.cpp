// TrustNav-IMU (C++17 port of src/main.py, same algorithm and fitted parameters).
// 6-DoF trajectory from a degraded IMU stream: EuRoC imu0/data.csv in, TUM poses out.
// Single pass, single thread, constant memory (~1 MB), no dependencies beyond the C++ standard library.
//
// Build: g++ -O2 -std=c++17 -o trustnav src/main.cpp
// Run  : ./trustnav imu0/data.csv estimated_trajectory.txt [--blind 5] [--no-fill]
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const double G0 = 9.80665;

struct Cfg {  // defaults = parameters fitted on the training flights (see tools/fit_params.py)
    double blind = 5.0, init_s = 1.0, init_search_s = 10.0;
    double spike_k = 8.0, kp = 1.0, ki = 0.005, tilt_gate = 3.0;
    double tau_v = 1.0, tau_vz = 0.5, tau_p = 3.0, tau_b = 0.5, tau_w = 3.0;
    double drag_k = 0.17, drag_beta = 1.0, tau_f = 0.3, tau_bias = 5.0;
    double zupt_a = 0.15, zupt_w = 0.05, vmax = 3.0, kp_gap = 1.0, relevel_s = 3.0;
};

struct Q { double w, x, y, z; };
static Q qmul(const Q& a, const Q& b) {
    return {a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z, a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x, a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w};
}
static Q qexp(double wx, double wy, double wz, double dt) {
    double n = std::sqrt(wx * wx + wy * wy + wz * wz), th = n * dt;
    if (th < 1e-12) return {1.0, 0.5 * wx * dt, 0.5 * wy * dt, 0.5 * wz * dt};
    double s = std::sin(0.5 * th) / n;
    return {std::cos(0.5 * th), wx * s, wy * s, wz * s};
}
static Q qnorm(const Q& q) {
    double n = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    return {q.w / n, q.x / n, q.y / n, q.z / n};
}
static void rot(const Q& q, const double v[3], double o[3]) {  // body -> world
    double tx = 2 * (q.y * v[2] - q.z * v[1]), ty = 2 * (q.z * v[0] - q.x * v[2]), tz = 2 * (q.x * v[1] - q.y * v[0]);
    o[0] = v[0] + q.w * tx + q.y * tz - q.z * ty;
    o[1] = v[1] + q.w * ty + q.z * tx - q.x * tz;
    o[2] = v[2] + q.w * tz + q.x * ty - q.y * tx;
}
static void rot_inv(const Q& q, const double v[3], double o[3]) { Q c{q.w, -q.x, -q.y, -q.z}; rot(c, v, o); }

struct Row { int64_t t; double v[6]; bool ok; };

static bool parse_row(const char* line, Row& r) {
    if (!line[0] || line[0] == '#' || std::isalpha((unsigned char)line[0])) return false;
    char* e;
    if (std::strchr(line, '.') && std::strchr(line, '.') < std::strchr(line, ','))
        r.t = (int64_t)std::llround(std::strtod(line, &e) * 1e9);
    else
        r.t = std::strtoll(line, &e, 10);
    const char* p = e;
    bool nz = false, fin = true;
    for (int i = 0; i < 6; i++) {
        if (*p != ',') return false;
        r.v[i] = std::strtod(p + 1, &e);
        if (e == p + 1) return false;
        p = e;
        if (!(r.v[i] == r.v[i]) || std::fabs(r.v[i]) >= 1e4) fin = false;
        if (r.v[i] != 0.0) nz = true;
    }
    r.ok = fin && nz;
    return true;
}

struct TrustNav {
    Cfg c;
    Q q{1, 0, 0, 0};
    double v[3] = {0, 0, 0}, p[3] = {0, 0, 0}, bg[3] = {0, 0, 0}, g = G0;
    double lp[6] = {0}, nl[6] = {0}, base[6], trust[6];
    double w_last[3] = {0, 0, 0}, n_b[3] = {1, 0, 0}, f_lp[3] = {0, 0, 0}, f_long[3] = {0, 0, 0};
    double gap_t = 0, za = G0, zv = 1, zw = 1, dt_nom = 0.005;
    long spikes = 0;
    bool is_static = false;

    explicit TrustNav(const Cfg& cfg) : c(cfg) { for (int i = 0; i < 6; i++) { base[i] = 1e-3; trust[i] = 1; } }

    void initialise(std::vector<Row>& all) {
        std::vector<Row> good;
        for (auto& r : all) if (r.ok) good.push_back(r);
        if (good.empty()) good = all;
        size_t n = good.size();
        std::vector<double> dts;
        for (size_t i = 0; i + 1 < n && i < 400; i++) dts.push_back((good[i + 1].t - good[i].t) * 1e-9);
        if (!dts.empty()) { std::sort(dts.begin(), dts.end()); dt_nom = std::max(1e-4, dts[dts.size() / 2]); }
        size_t W = std::max<size_t>(20, (size_t)std::llround(c.init_s / dt_nom));
        double best = 1e300; size_t s0b = 0; double gsb = 0, asb = 0;
        size_t lim = n >= W ? n - W + 1 : 1, stepw = std::max<size_t>(1, W / 4);
        for (size_t s0 = 0; s0 < lim; s0 += stepw) {
            size_t m = std::min(W, n - s0);
            double mw[3] = {0, 0, 0}, am = 0;
            for (size_t k = s0; k < s0 + m; k++) for (int i = 0; i < 3; i++) mw[i] += good[k].v[i] / m;
            double gs = 0, as = 0;
            for (size_t k = s0; k < s0 + m; k++) {
                for (int i = 0; i < 3; i++) gs += (good[k].v[i] - mw[i]) * (good[k].v[i] - mw[i]) / m;
                am += std::sqrt(good[k].v[3] * good[k].v[3] + good[k].v[4] * good[k].v[4] + good[k].v[5] * good[k].v[5]) / m;
            }
            for (size_t k = s0; k < s0 + m; k++) {
                double a = std::sqrt(good[k].v[3] * good[k].v[3] + good[k].v[4] * good[k].v[4] + good[k].v[5] * good[k].v[5]);
                as += (a - am) * (a - am) / m;
            }
            gs = std::sqrt(gs); as = std::sqrt(as);
            double sc = gs / 0.02 + as / 0.1;
            if (sc < best) { best = sc; s0b = s0; gsb = gs; asb = as; }
        }
        size_t m = std::min(W, n - s0b);
        double mw[3] = {0, 0, 0}, ma[3] = {0, 0, 0};
        for (size_t k = s0b; k < s0b + m; k++) for (int i = 0; i < 3; i++) { mw[i] += good[k].v[i] / m; ma[i] += good[k].v[3 + i] / m; }
        double an = std::sqrt(ma[0] * ma[0] + ma[1] * ma[1] + ma[2] * ma[2]);
        is_static = gsb < 0.05 && asb < 0.15;
        double u[3] = {ma[0] / an, ma[1] / an, ma[2] / an};
        if (is_static) { for (int i = 0; i < 3; i++) bg[i] = mw[i]; g = an; }
        Q qq = u[2] < -0.999999 ? Q{0, 1, 0, 0} : qnorm(Q{1 + u[2], u[1], -u[0], 0});
        for (size_t i = s0b + m / 2; i > 0; i--) {  // carry attitude back to the first frame
            double dt = (good[i].t - good[i - 1].t) * 1e-9;
            const double* w = good[i - 1].v;
            qq = qnorm(qmul(qq, qexp(-(w[0] - bg[0]), -(w[1] - bg[1]), -(w[2] - bg[2]), dt)));
        }
        q = qq;
        for (int i = 0; i < 3; i++) { n_b[i] = u[i]; f_lp[i] = good[0].v[3 + i]; f_long[i] = ma[i]; }
        std::vector<double> d(m);
        for (int i = 0; i < 6; i++) {
            double mm = 0;
            for (size_t k = s0b; k < s0b + m; k++) mm += good[k].v[i] / m;
            for (size_t k = 0; k < m; k++) d[k] = std::fabs(good[s0b + k].v[i] - mm);
            std::sort(d.begin(), d.end());
            base[i] = std::max(d[m / 2] * 1.4826, 1e-5);
            nl[i] = base[i];
            lp[i] = good[0].v[i];
        }
    }

    void clean(double v_[6]) {
        for (int i = 0; i < 6; i++) {
            double ad = std::fabs(v_[i] - lp[i]);
            double lim = c.spike_k * std::max(nl[i], base[i]);
            if (ad > lim && ad > (i < 3 ? 0.6 : 6.0)) { v_[i] = lp[i]; spikes++; ad = lim; }
            nl[i] += 0.01 * (ad - nl[i]);
            lp[i] += 0.3 * (v_[i] - lp[i]);
            double r = base[i] / std::max(nl[i], 1e-9);
            trust[i] = r < 1 ? r : 1;
        }
    }

    double acc_trust() const { return std::min(trust[3], std::min(trust[4], trust[5])); }

    void step(double dt, const double m[6]) {
        double w[3] = {m[0] - bg[0], m[1] - bg[1], m[2] - bg[2]}, a[3] = {m[3], m[4], m[5]};
        double an = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
        if (std::fabs(an - g) < c.tilt_gate) {  // trust-weighted tilt correction
            double up[3] = {0, 0, 1}, ub[3];
            rot_inv(q, up, ub);
            double f[3] = {a[0] / an, a[1] / an, a[2] / an};
            double e[3] = {f[1] * ub[2] - f[2] * ub[1], f[2] * ub[0] - f[0] * ub[2], f[0] * ub[1] - f[1] * ub[0]};
            double tg = acc_trust(), k = c.kp * tg;
            if (gap_t > 0) { gap_t -= dt; k = std::max(k, c.kp_gap); }
            double ki = c.ki * tg * dt;
            for (int i = 0; i < 3; i++) { w[i] += k * e[i]; bg[i] -= ki * e[i]; }
        }
        Q q1 = qnorm(qmul(q, qexp(w[0], w[1], w[2], dt)));
        double fa[3], fb[3];
        rot(q, a, fa); rot(q1, a, fb);
        double aw[3] = {0.5 * (fa[0] + fb[0]), 0.5 * (fa[1] + fb[1]), 0.5 * (fa[2] + fb[2]) - g};
        q = q1;
        double nv[3];
        for (int i = 0; i < 3; i++) nv[i] = v[i] + aw[i] * dt;
        double dec = std::exp(-dt / c.tau_v);  // learned motion prior
        nv[0] *= dec; nv[1] *= dec; nv[2] *= std::exp(-dt / c.tau_vz);
        double np_[3];
        for (int i = 0; i < 3; i++) np_[i] = p[i] + 0.5 * (v[i] + nv[i]) * dt;
        double kk = dt / c.tau_p;
        for (int i = 0; i < 3; i++) np_[i] -= kk * np_[i];
        drag_update(dt, a, nv);
        e_zupt(dt, an, w);
        if (zv < c.zupt_a * c.zupt_a && zw < c.zupt_w && std::fabs(za - g) < 0.3) {
            nv[0] = nv[1] = nv[2] = 0; for (int i = 0; i < 3; i++) np_[i] = p[i];
        }
        double sp = std::sqrt(nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2]);
        if (sp > c.vmax) for (int i = 0; i < 3; i++) nv[i] *= c.vmax / sp;
        {   // physical guardrail: the position may not move faster than the speed bound
            double d[3] = {np_[0] - p[0], np_[1] - p[1], np_[2] - p[2]};
            double dn = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]), lim = c.vmax * dt;
            if (dn > lim) for (int i = 0; i < 3; i++) np_[i] = p[i] + d[i] * (lim / dn);
        }
        for (int i = 0; i < 3; i++) { v[i] = nv[i]; p[i] = np_[i]; w_last[i] = 0.98 * w_last[i] + 0.02 * w[i]; }
    }

    void e_zupt(double dt, double an, const double w[3]) {
        double e = dt / 0.3;
        za += e * (an - za);
        zv += e * ((an - za) * (an - za) - zv);
        zw += e * (std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]) - zw);
    }

    void drag_update(double dt, const double f[3], double vw[3]) {  // rotor-drag velocity aiding
        double a = dt / c.tau_f, b2 = dt / c.tau_bias, d[3];
        for (int i = 0; i < 3; i++) {
            f_lp[i] += a * (f[i] - f_lp[i]);
            f_long[i] += b2 * (f[i] - f_long[i]);
            d[i] = f_lp[i] - f_long[i];
        }
        double dn = d[0] * n_b[0] + d[1] * n_b[1] + d[2] * n_b[2];
        double vb[3]; rot_inv(q, vw, vb);
        double vn = vb[0] * n_b[0] + vb[1] * n_b[1] + vb[2] * n_b[2];
        double b = std::min(1.0, c.drag_beta * acc_trust() * dt);
        for (int i = 0; i < 3; i++) {
            double fp = d[i] - dn * n_b[i], vp = vb[i] - vn * n_b[i];
            vp += b * (-fp / c.drag_k - vp);
            vb[i] = vp + vn * n_b[i];
        }
        rot(q, vb, vw);
    }

    void coast(double dt) {  // no IMU data: dropout or failure
        gap_t = c.relevel_s;
        double dec = std::exp(-dt / c.tau_b);
        v[0] *= dec; v[1] *= dec; v[2] *= std::exp(-dt / c.tau_vz);
        double wd = std::exp(-dt / c.tau_w);
        for (int i = 0; i < 3; i++) w_last[i] *= wd;
        double up[3] = {0, 0, 1}, u[3]; rot_inv(q, up, u);
        double wz = w_last[0] * u[0] + w_last[1] * u[1] + w_last[2] * u[2];
        q = qnorm(qmul(q, qexp(wz * u[0], wz * u[1], wz * u[2], dt)));
        for (int i = 0; i < 3; i++) p[i] += v[i] * dt;
    }

    void write(FILE* f, int64_t tns) const {
        std::fprintf(f, "%lld.%09lld %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n", (long long)(tns / 1000000000LL),
                     (long long)(tns % 1000000000LL), p[0], p[1], p[2], q.x, q.y, q.z, q.w);
    }
};

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s imu0/data.csv estimated_trajectory.txt [--blind S] [--no-fill]\n", argv[0]); return 2; }
    Cfg cfg;
    bool fill = true;  // --no-fill: write poses only at input IMU timestamps (+ blind window)
    for (int i = 3; i < argc; i++) {
        if (!std::strcmp(argv[i], "--blind") && i + 1 < argc) cfg.blind = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--no-fill")) fill = false;
        else { std::fprintf(stderr, "unknown option: %s\n", argv[i]); return 2; }
    }
    FILE* in = std::fopen(argv[1], "r");
    FILE* out = std::fopen(argv[2], "w");
    if (!in || !out) { std::fprintf(stderr, "cannot open files\n"); return 1; }
    std::fputs("# timestamp tx ty tz qx qy qz qw\n", out);
    TrustNav est(cfg);
    char line[512];
    std::vector<Row> buf;
    Row r;
    while (std::fgets(line, sizeof line, in)) {  // bounded buffer for initialisation
        if (!parse_row(line, r)) continue;
        buf.push_back(r);
        if ((r.t - buf[0].t) * 1e-9 >= cfg.init_search_s || buf.size() >= 4000) break;
    }
    if (buf.empty()) { std::fclose(out); return 0; }
    est.initialise(buf);
    int64_t last = buf[0].t, dtn = (int64_t)std::llround(est.dt_nom * 1e9);
    est.write(out, last);
    auto handle = [&](Row& rr) {
        int64_t d = rr.t - last;
        if (d <= 0) return;
        if (fill && d > (int64_t)(1.6 * dtn)) {  // gap: fill at the nominal rate
            long k = std::lround((double)d / dtn) - 1;
            for (long j = 0; j < k; j++) { last += dtn; est.coast(dtn * 1e-9); est.write(out, last); }
            d = rr.t - last;
            if (d <= 0) return;
        }
        if (rr.ok) { est.clean(rr.v); est.step(d * 1e-9, rr.v); }
        else est.coast(d * 1e-9);
        est.write(out, rr.t);
        last = rr.t;
    };
    for (size_t i = 1; i < buf.size(); i++) handle(buf[i]);
    std::vector<Row>().swap(buf);
    while (std::fgets(line, sizeof line, in))
        if (parse_row(line, r)) handle(r);
    long nb = std::lround(cfg.blind / est.dt_nom);  // IMU has failed: blind dead-reckoning
    for (long j = 0; j < nb; j++) { last += dtn; est.coast(dtn * 1e-9); est.write(out, last); }
    std::fclose(in); std::fclose(out);
    std::fprintf(stderr, "spikes rejected: %ld, static start: %d, g=%.4f\n", est.spikes, (int)est.is_static, est.g);
    if (FILE* st = std::fopen("/proc/self/status", "r")) {  // Linux: report peak resident memory
        char l[256];
        while (std::fgets(l, sizeof l, st)) if (!std::strncmp(l, "VmHWM", 5)) std::fprintf(stderr, "peak %s", l);
        std::fclose(st);
    }
    return 0;
}
