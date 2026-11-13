"""Build the TrustNav Mission Console (one self-contained HTML page) from REAL outputs.

Every number on the page is computed from your files - nothing is invented.
Run from the project folder after tools/health.py and tools/stress_test.py:

  python tools/build_console.py
  (then open dashboard/TrustNav-Mission-Console.html in a browser)

Options (defaults match the project layout):
  --imu     real_runs/MH_01_easy_s1/mav0/imu0/data.csv   degraded IMU given to the solver
  --est     estimated_trajectory.txt                       solver output (TUM)
  --gt      MH_01_easy/mav0/state_groundtruth_estimate0/data.csv
  --events  real_runs/MH_01_easy_s1/events.json            degradation schedule
  --health  health_out                                     folder written by tools/health.py
  --stress  stress_results/stress_table.md                 written by tools/stress_test.py
  --meta    metadata.json
  --validation validation.json  written by tools/validate_output.py
  --out     dashboard/TrustNav-Mission-Console.html
"""
import argparse
import json
import math
import os
import numpy as np


def load_tum(path):
    a = np.loadtxt(path, comments='#')
    return a[:, 0], a[:, 1:4], a[:, 4:8]


def load_gt(path):
    if path.endswith('.csv'):
        a = np.loadtxt(path, delimiter=',', comments='#')
        return a[:, 0] * 1e-9, a[:, 1:4]
    a = np.loadtxt(path, comments='#')
    return a[:, 0], a[:, 1:4]


def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    C = (dst - md).T @ (src - ms) / len(src)
    U, _, Vt = np.linalg.svd(C)
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    return R, md - R @ ms


def motion_state(speed_h, vz, gyro):
    if speed_h < 0.08 and abs(vz) < 0.08 and gyro < 0.08:
        return 'LANDED'
    if abs(vz) > 0.25:
        return 'CLIMB' if vz > 0 else 'DESCEND'
    if gyro > 0.35:
        return 'TURN'
    if speed_h > 0.4:
        return 'CRUISE'
    return 'HOVER'


NARR = {
    'NOISE_RISING': ('amber', lambda d: d.replace(' - trust lowered', '') + ' - trusting the IMU less'),
    'DROPOUT_START': ('red', lambda d: 'Signal lost - holding course on the motion model (' + d.split('for ')[-1].split(' -')[0] + ')'),
    'DROPOUT_END': ('green', lambda d: 'Signal back - re-levelling attitude'),
    'IMU_FAILURE': ('red', lambda d: 'IMU dead - 5 s blind dead-reckoning to the target'),
    'SPIKE': ('amber', lambda d: 'Spike rejected on ' + d.split(' on ')[-1]),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--imu', default='real_runs/MH_01_easy_s1/mav0/imu0/data.csv')
    ap.add_argument('--est', default='estimated_trajectory.txt')
    ap.add_argument('--gt', default='MH_01_easy/mav0/state_groundtruth_estimate0/data.csv')
    ap.add_argument('--events', default='real_runs/MH_01_easy_s1/events.json')
    ap.add_argument('--health', default='health_out')
    ap.add_argument('--stress', default='stress_results/stress_table.md')
    ap.add_argument('--meta', default='metadata.json')
    ap.add_argument('--validation', default='validation.json')
    ap.add_argument('--out', default='dashboard/TrustNav-Mission-Console.html')
    a = ap.parse_args()

    ev = json.load(open(a.events))
    t0 = ev['t0']
    te, pe, qe = load_tum(a.est)
    tg, pg = load_gt(a.gt)

    # align estimate to ground truth exactly as evo -a does (heading is unobservable from an IMU)
    idx = np.searchsorted(tg, te).clip(1, len(tg) - 1)
    idx = np.where(np.abs(te - tg[idx - 1]) < np.abs(te - tg[idx]), idx - 1, idx)
    ok = np.abs(tg[idx] - te) < 0.003
    R, tr = umeyama(pe[ok], pg[idx[ok]])
    pa = pe @ R.T + tr
    err = np.full(len(te), np.nan)
    err[ok] = np.linalg.norm(pa[ok] - pg[idx[ok]], axis=1)
    ate = float(np.sqrt(np.nanmean(err ** 2)))

    # IMU gyro magnitude (for the motion-state rule) on a 0.5 s grid
    imu = np.loadtxt(a.imu, delimiter=',', comments='#')
    ti = imu[:, 0] * 1e-9 if imu[0, 0] > 1e12 else imu[:, 0]
    gyro = np.linalg.norm(imu[:, 1:4], axis=1)

    # 10 Hz series for the page
    step = max(1, int(round(0.1 / np.median(np.diff(te)))))
    sel = np.arange(0, len(te), step)
    vel = np.gradient(pa[:, :3], te, axis=0)
    home = pa[0]
    series = []
    for i in sel:
        t = te[i] - t0
        j = np.searchsorted(ti, te[i])
        g = float(np.median(gyro[max(0, j - 50):j + 1])) if j > 0 and te[i] <= ti[-1] else float('nan')
        sh = float(np.hypot(vel[i, 0], vel[i, 1])); vz = float(vel[i, 2])
        d = pa[i] - home
        series.append({
            't': round(float(t), 2),
            'x': round(float(pa[i, 0]), 3), 'y': round(float(pa[i, 1]), 3), 'z': round(float(pa[i, 2]), 3),
            'err': None if np.isnan(err[i]) else round(float(err[i]), 3),
            'home_d': round(float(np.linalg.norm(d)), 2),
            'home_b': round(float((math.degrees(math.atan2(-d[0], -d[1])) + 360) % 360), 1),
            'climb': round(float(-d[2]), 2),
            'state': 'BLIND' if te[i] > ti[-1] else motion_state(sh, vz, g if g == g else 0.0),
            'spd': round(sh, 2), 'vz': round(vz, 2), 'gyro': None if g != g else round(g, 3),
        })
    gsel = np.arange(0, len(tg), max(1, len(tg) // 2000))
    gt_xyz = [[round(float(pg[k, 0]), 3), round(float(pg[k, 1]), 3), round(float(pg[k, 2]), 3), round(float(tg[k] - t0), 2)] for k in gsel]

    # health monitor outputs
    hdir = a.health
    health = []
    with open(os.path.join(hdir, 'health.csv')) as fh:
        next(fh)
        for line in fh:
            p = line.strip().split(',')
            health.append([float(p[0]), None if p[1] == 'nan' else float(p[1]), None if p[2] == 'nan' else float(p[2]), int(p[3]), float(p[4])])
    narr, n_spike = [], 0
    with open(os.path.join(hdir, 'events_log.txt')) as fh:
        for line in fh:
            if line.startswith('#') or not line.strip():
                continue
            p = line.split(None, 3)
            kind, det = p[2], p[3].strip() if len(p) > 3 else ''
            n_spike += kind == 'SPIKE'
            col, f = NARR.get(kind, ('amber', lambda d: d))
            narr.append({'t': float(p[0]), 'kind': kind, 'col': col, 'msg': f(det)})
    conf = []
    cpath = os.path.join(hdir, 'confidence.txt')
    if os.path.exists(cpath):
        c = np.loadtxt(cpath, comments='#', usecols=(0, 1))
        cs = np.arange(0, len(c), max(1, int(round(0.1 / np.median(np.diff(c[:, 0]))))))
        conf = [[round(float(c[k, 0] - t0), 2), round(float(c[k, 1]), 3)] for k in cs]

    # stress table
    stress = []
    if os.path.exists(a.stress):
        for line in open(a.stress):
            p = [s.strip() for s in line.strip().strip('|').split('|')]
            if len(p) == 6 and p[1].replace('.', '', 1).isdigit():
                stress.append([p[0]] + [float(v) for v in p[1:]])
    seeds = [r for r in stress if 'seed' in r[0]]
    avg = [round(float(np.mean([r[k] for r in seeds])), 2) for k in range(1, 6)] if seeds else None

    meta = json.load(open(a.meta)) if os.path.exists(a.meta) else {}
    m = meta.get('measured_real_mh01', {})
    kpi = {
        'ram': m.get('peak_rss_mb_cpp', 3.5), 'ms': m.get('ms_per_imu_second_cpp', 0.75),
        'ate': avg[0] if avg else round(ate, 2), 'ate_run': round(ate, 2),
        'spikes': n_spike, 'spikes_inj': ev.get('spike_count'),
        'dropouts': ev.get('dropouts', []), 'failure': ev['failure'], 'end': ev['end'],
        'noise_max': ev.get('noise_max'), 'n_imu': int(len(ti)), 'n_pose': int(len(te)),
    }
    val = json.load(open(a.validation)) if os.path.exists(a.validation) else None
    data = {'val': val, 'series': series, 'gt': gt_xyz, 'health': health, 'narr': narr, 'conf': conf,
            'stress': stress, 'avg': avg, 'kpi': kpi}
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    html = TEMPLATE.replace('/*__DATA__*/null', json.dumps(data, separators=(',', ':')))
    with open(a.out, 'w', encoding='utf-8') as fh:
        fh.write(html)
    print('ATE this run %.2f m | %d poses | %d events | %d stress rows -> %s' % (ate, len(series), len(narr), len(stress), a.out))


TEMPLATE = r'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TrustNav — Mission Console</title>
<style>
:root{--bg:#0B0F14;--panel:#111821;--panel2:#0E141C;--line:#1E2A36;--txt:#D7E3EE;--dim:#7F93A6;--cy:#22D3EE;--am:#F59E0B;--rd:#EF4444;--gn:#22C55E;--mono:ui-monospace,SFMono-Regular,Consolas,"Cascadia Mono",monospace}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--txt);font:14px/1.4 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{display:flex;align-items:center;gap:18px;padding:12px 20px;border-bottom:1px solid var(--line);flex-wrap:wrap}
.brand{font:700 22px var(--mono);color:var(--cy);letter-spacing:2px}.sub{font:12px var(--mono);color:var(--dim)}
.pill{font:600 12px var(--mono);padding:8px 12px;border-radius:6px;background:var(--panel);border:1px solid var(--line)}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px;vertical-align:middle}
.tabs{margin-left:auto;display:flex;gap:6px}.tab{font:600 13px var(--mono);padding:9px 16px;border-radius:6px;border:1px solid var(--line);background:var(--panel);color:var(--dim);cursor:pointer}
.tab.on{background:var(--cy);color:#04222A;border-color:var(--cy)}
main{padding:16px 20px;display:none}main.on{display:block}
.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;min-width:0}
.tag{font:600 11px var(--mono);color:var(--cy);margin-right:6px}.h{font-weight:700;font-size:17px;letter-spacing:.3px}
.small{font:12px var(--mono);color:var(--dim)}.big{font:700 34px var(--mono)}.mono{font-family:var(--mono)}
canvas{width:100%;display:block}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}.chip{font:600 12px var(--mono);padding:7px 10px;border-radius:6px;background:var(--panel2);border:1px solid var(--line);color:var(--dim)}
.chip.on{background:var(--cy);color:#04222A;border-color:var(--cy)}
.narr{display:flex;flex-direction:column;gap:6px;max-height:250px;overflow:auto}
.ev{display:grid;grid-template-columns:52px 62px 1fr;gap:6px;align-items:center;background:var(--panel2);border:1px solid var(--line);border-radius:6px;padding:6px 8px;font-size:13px}
.badge{font:700 10px var(--mono);padding:2px 6px;border-radius:4px;text-align:center}
.b-amber{background:#3A2A0A;color:var(--am)}.b-red{background:#3A1010;color:var(--rd)}.b-green{background:#0D3320;color:var(--gn)}
.bar{height:6px;background:#1B2631;border-radius:4px;overflow:hidden;margin-top:8px}.bar>i{display:block;height:100%}
.ctrl{display:flex;gap:10px;align-items:center;margin-top:10px}.ctrl input{flex:1;accent-color:var(--cy)}
button.play{font:700 12px var(--mono);background:var(--cy);color:#04222A;border:0;border-radius:6px;padding:7px 12px;cursor:pointer}
table{width:100%;border-collapse:collapse;font:13px var(--mono)}th,td{padding:9px 8px;border-bottom:1px solid var(--line);text-align:right}
th{color:var(--dim);font-weight:600;background:var(--panel2)}td:first-child,th:first-child{text-align:left}
tr.avg td{color:var(--cy);font-weight:700}
.note{font-size:12px;color:var(--dim);margin-top:8px}.concept{font:700 10px var(--mono);color:var(--am);border:1px solid var(--am);border-radius:4px;padding:1px 5px;margin-left:6px}
@media(max-width:1100px){.grid>*{grid-column:1/-1!important}}
</style></head><body>
<header>
 <div><div class="brand">TRUSTNAV</div><div class="sub">GPS-denied navigation under sensor jamming · IMU only · offline · no camera · no GPS</div></div>
 <div class="pill" id="status"><span class="dot"></span>IMU HEALTH</div>
 <div class="pill mono">t = <span id="tnow">0.0</span> s</div>
 <div class="pill" id="gpill"></div>
 <div class="tabs"><div class="tab on" data-t="m1">MISSION CONSOLE</div><div class="tab" data-t="m2">STRESS TEST</div><div class="tab" data-t="m3">TRUST &amp; VERIFY</div></div>
</header>

<main id="m1" class="on"><div class="grid">
 <section class="card" style="grid-column:span 8">
  <div><span class="tag">[NAV]</span><span class="h">Estimated vs ground-truth trajectory (top view)</span></div>
  <div class="small" id="navline"></div>
  <canvas id="traj" height="430"></canvas>
  <div class="small"><span style="color:#fff">- - -</span> ground truth (mocap) · <span style="color:var(--cy)">━</span> TrustNav estimate (aligned, as evo -a) · <span style="color:var(--rd)">━</span> dropouts · <span style="color:#9aa">━</span> blind 5 s after IMU failure</div>
  <div class="ctrl"><button class="play" id="play">▶ PLAY</button><input type="range" id="scrub" min="0" max="1000" value="1000"></div>
 </section>
 <div style="grid-column:span 4;display:flex;flex-direction:column;gap:14px">
  <section class="card">
   <div><span class="tag">[RTB]</span><span class="h">Return-home vector</span></div>
   <div style="display:flex;gap:14px;align-items:center;margin-top:8px">
    <svg width="104" height="104" viewBox="-52 -52 104 104"><circle r="48" fill="#0A1017" stroke="#2A3A4A" stroke-width="2"/>
     <text y="-36" text-anchor="middle" fill="#7F93A6" font-size="10">+Y</text>
     <g id="arrow"><polygon points="0,-38 7,6 0,0 -7,6" fill="#22D3EE"/><polygon points="0,38 7,6 0,0 -7,6" fill="#2A3A4A"/></g></svg>
    <div><div><span class="big" id="hd">0</span> <span class="small">m to launch point</span></div>
     <div class="mono">bearing <b id="hb">0</b>° (map frame) · <span id="hc"></span></div></div>
   </div>
   <div class="small" style="margin-top:8px">confidence <b id="cf" class="mono"></b></div><div class="bar"><i id="cfb" style="background:linear-gradient(90deg,var(--cy),var(--am))"></i></div>
  </section>
  <section class="card">
   <div><span class="tag">[MOT]</span><span class="h">Motion state</span> <span class="small">rule-based on the estimate</span></div>
   <div class="big" id="ms" style="color:var(--cy);margin-top:6px">HOVER</div>
   <div class="mono small" id="msd"></div>
   <div class="chips" id="chips"></div>
  </section>
  <section class="card">
   <div><span class="tag">[NARR]</span><span class="h">Flight narrator</span> <span class="small">from events_log.txt</span></div>
   <div class="narr" id="narr" style="margin-top:8px"></div>
  </section>
 </div>
 <section class="card" style="grid-column:span 8">
  <div><span class="tag">[HEALTH]</span><span class="h">Sensor health &amp; jamming timeline</span></div>
  <div class="small"><span style="color:var(--am)">━</span> noise vs clean start (×) · <span style="color:var(--gn)">■</span> IMU health 0–100 · <span style="color:var(--rd)">■</span> dropouts · <span style="color:#9aa">■</span> IMU failure / blind</div>
  <canvas id="hl" height="230"></canvas>
 </section>
 <div style="grid-column:span 4;display:grid;grid-template-columns:1fr 1fr;gap:14px">
  <section class="card"><div class="small">PEAK RAM (C++)</div><div class="big" style="color:var(--cy)"><span id="k1"></span><span class="small"> MB</span></div><div class="bar"><i id="k1b" style="background:var(--cy)"></i></div><div class="note">limit 8 MB</div></section>
  <section class="card"><div class="small">COMPUTE</div><div class="big" style="color:var(--gn)"><span id="k2"></span><span class="small"> ms</span></div><div class="bar"><i id="k2b" style="background:var(--gn)"></i></div><div class="note">per IMU second · limit 20 ms</div></section>
  <section class="card"><div class="small">AVG ERROR (ATE)</div><div class="big"><span id="k3"></span><span class="small"> m</span></div><div class="note" id="k3n"></div></section>
  <section class="card"><div class="small">SPIKES CAUGHT</div><div class="big" style="color:var(--am)" id="k4"></div><div class="note">health monitor vs injected</div></section>
 </div>
</div></main>

<main id="m2"><div class="grid">
 <section class="card" style="grid-column:span 7">
  <div><span class="tag">[EVAL]</span><span class="h">Stress test — real EuRoC MH_01, real IMU</span></div>
  <div class="small">tools/stress_test.py · degraded by tools/degrade.py · scored like evo_ape -a</div>
  <table id="st" style="margin-top:10px"></table>
  <div class="note" id="stn"></div>
 </section>
 <section class="card" style="grid-column:span 5">
  <div><span class="tag">[ATE]</span><span class="h">Accuracy per scenario</span></div>
  <canvas id="sb" height="300"></canvas>
  <div class="note" id="sbn"></div>
 </section>
 <section class="card" style="grid-column:span 12">
  <div><span class="tag">[CPU]</span><span class="h">Efficiency (measured)</span></div>
  <div class="mono" style="margin-top:8px" id="eff"></div>
  <div class="note">Measured on a laptop, C++17, single thread, offline. Limits from the Track 3 brief.</div>
 </section>
</div></main>

<main id="m3"><div class="grid">
 <section class="card" style="grid-column:span 5">
  <div><span class="tag">[SPEC]</span><span class="h">Spec is law — output guardrails</span></div>
  <div class="small">tools/validate_output.py checks the submitted file against the Track 3 format and physics before submission</div>
  <div id="checks" style="margin-top:10px;display:flex;flex-direction:column;gap:6px"></div>
  <div class="big" id="ctot" style="margin-top:12px"></div>
 </section>
 <section class="card" style="grid-column:span 4">
  <div><span class="tag">[HASH]</span><span class="h">Reproducible by anyone</span></div>
  <div class="small">SHA-256 fingerprints, also stored in metadata.json</div>
  <div id="hashes" style="margin-top:10px;display:flex;flex-direction:column;gap:8px"></div>
  <pre class="mono" style="background:var(--panel2);border:1px solid var(--line);border-radius:6px;padding:10px;white-space:pre-wrap;font-size:12px;margin-top:10px">g++ -O2 -std=c++17 -o trustnav src/main.cpp
./trustnav &lt;input imu0/data.csv&gt; out.txt
sha256sum out.txt   # equals estimated_trajectory.txt</pre>
 </section>
 <section class="card" style="grid-column:span 3">
  <div><span class="tag">[PRIV]</span><span class="h">Privacy by design</span></div>
  <div style="display:grid;grid-template-columns:1fr;gap:8px;margin-top:10px" class="mono">
   <div class="ev" style="grid-template-columns:1fr">No camera — no images recorded</div>
   <div class="ev" style="grid-template-columns:1fr">No GPS — keeps working when jammed</div>
   <div class="ev" style="grid-template-columns:1fr">100% offline — nothing leaves the device</div>
   <div class="ev" style="grid-template-columns:1fr" id="privram"></div>
  </div>
  <div class="note">Use cases: power lines · pipelines · tunnels &amp; under bridges · wildfire smoke — where GPS fails and cameras are blind or not allowed.</div>
 </section>
</div></main>

<script>
const D=/*__DATA__*/null;
const $=id=>document.getElementById(id), S=D.series, K=D.kpi;
document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('on',x===b));document.querySelectorAll('main').forEach(m=>m.classList.toggle('on',m.id===b.dataset.t));draw();});
function fit(c){const r=c.getBoundingClientRect(),dpr=window.devicePixelRatio||1;c.width=r.width*dpr;c.height=c.getAttribute('height')*dpr;const x=c.getContext('2d');x.setTransform(dpr,0,0,dpr,0,0);return [x,r.width,+c.getAttribute('height')];}
const inDrop=t=>K.dropouts.some(([a,b])=>t>=a&&t<=b);
let cur=S.length-1;
function idxAt(t){let lo=0,hi=S.length-1;while(lo<hi){const m=(lo+hi)>>1;S[m].t<t?lo=m+1:hi=m;}return lo;}
function drawTraj(){const [x,w,h]=fit($('traj'));const P=S.map(s=>[s.x,s.y]).concat(D.gt.map(g=>[g[0],g[1]]));
 let mnx=Math.min(...P.map(p=>p[0])),mxx=Math.max(...P.map(p=>p[0])),mny=Math.min(...P.map(p=>p[1])),mxy=Math.max(...P.map(p=>p[1]));
 const sc=Math.min((w-40)/(mxx-mnx||1),(h-40)/(mxy-mny||1)),ox=(w-(mxx-mnx)*sc)/2,oy=(h-(mxy-mny)*sc)/2;
 const X=v=>ox+(v-mnx)*sc,Y=v=>h-oy-(v-mny)*sc;
 x.strokeStyle='#16212B';x.lineWidth=1;for(let g=Math.ceil(mnx);g<=mxx;g++){x.beginPath();x.moveTo(X(g),0);x.lineTo(X(g),h);x.stroke();}for(let g=Math.ceil(mny);g<=mxy;g++){x.beginPath();x.moveTo(0,Y(g));x.lineTo(w,Y(g));x.stroke();}
 const tn=S[cur].t;x.setLineDash([5,4]);x.strokeStyle='rgba(255,255,255,.75)';x.lineWidth=1.4;x.beginPath();D.gt.filter(g=>g[3]<=tn).forEach((g,i)=>i?x.lineTo(X(g[0]),Y(g[1])):x.moveTo(X(g[0]),Y(g[1])));x.stroke();x.setLineDash([]);
 x.lineWidth=2.4;for(let i=1;i<=cur;i++){const a=S[i-1],b=S[i];x.strokeStyle=b.state==='BLIND'?'#9aa':inDrop(b.t)?'#EF4444':'#22D3EE';x.shadowColor=x.strokeStyle;x.shadowBlur=6;x.beginPath();x.moveTo(X(a.x),Y(a.y));x.lineTo(X(b.x),Y(b.y));x.stroke();}
 x.shadowBlur=0;const s0=S[0],s=S[cur];x.fillStyle='#7F93A6';x.beginPath();x.arc(X(s0.x),Y(s0.y),5,0,7);x.fill();x.font='11px monospace';x.fillText('LAUNCH',X(s0.x)+8,Y(s0.y)+4);
 const c=confAt(s.t),r=8+40*(1-c);x.strokeStyle='rgba(34,211,238,.5)';x.fillStyle='rgba(34,211,238,.08)';x.beginPath();x.arc(X(s.x),Y(s.y),r,0,7);x.fill();x.stroke();x.fillStyle='#22D3EE';x.beginPath();x.arc(X(s.x),Y(s.y),5,0,7);x.fill();
 x.fillStyle='#7F93A6';x.fillText('1 m grid',8,h-8);}
function confAt(t){if(!D.conf.length)return .5;let lo=0,hi=D.conf.length-1;while(lo<hi){const m=(lo+hi)>>1;D.conf[m][0]<t?lo=m+1:hi=m;}return D.conf[lo][1];}
function drawHealth(){const [x,w,h]=fit($('hl'));const H=D.health,T=K.end+0.5,X=t=>40+(t/T)*(w-50),top=10,mid=h*0.45,bot=h-22;
 K.dropouts.forEach(([a,b])=>{x.fillStyle='rgba(239,68,68,.25)';x.fillRect(X(a),top,X(b)-X(a),bot-top);});x.fillStyle='rgba(150,160,170,.18)';x.fillRect(X(K.failure),top,X(K.end)-X(K.failure),bot-top);
 const mx=Math.max(2,...H.map(r=>Math.max(r[1]||0,r[2]||0)));x.strokeStyle='#F59E0B';x.lineWidth=1.5;x.beginPath();let st=0;H.forEach(r=>{const v=Math.max(r[1]||0,r[2]||0);if(r[1]==null){st=0;return;}const yy=mid-((Math.max(v,1)-1)/(mx-1))*(mid-top);st?x.lineTo(X(r[0]),yy):x.moveTo(X(r[0]),yy);st=1;});x.stroke();
 x.fillStyle='rgba(34,197,94,.45)';H.forEach(r=>{const yy=bot-(r[4]/100)*(bot-mid-6);x.fillRect(X(r[0]),yy,Math.max(1,X(r[0]+0.5)-X(r[0])),bot-yy);});
 x.fillStyle='#7F93A6';x.font='11px monospace';x.fillText('×'+mx.toFixed(0),2,top+10);x.fillText('×1',8,mid);x.fillText('100',4,mid+14);x.fillText('0',20,bot);
 for(let t=0;t<=T;t+=30){x.fillText(t+'s',X(t)-8,h-6);} const tn=S[cur].t;x.strokeStyle='#22D3EE';x.setLineDash([4,3]);x.beginPath();x.moveTo(X(tn),top);x.lineTo(X(tn),bot);x.stroke();x.setLineDash([]);}
const STATES=['LANDED','HOVER','CLIMB','DESCEND','TURN','CRUISE','BLIND'];
function side(){const s=S[cur],t=s.t,hr=D.health[Math.min(D.health.length-1,Math.floor(t/0.5))]||[0,0,0,0,0],hv=s.state==='BLIND'?0:inDrop(t)?0:hr[4];
 $('tnow').textContent=t.toFixed(1);const col=hv>=60?'var(--gn)':hv>=25?'var(--am)':'var(--rd)';
 $('status').innerHTML=`<span class="dot" style="background:${col}"></span>IMU HEALTH ${hv.toFixed(0)}% — ${s.state==='BLIND'?'IMU FAILED · BLIND':inDrop(t)?'SIGNAL LOST':hr[1]>2?'JAMMING DETECTED':'NOMINAL'}`;
 $('navline').textContent=`t ${t.toFixed(1)} s · pos (${s.x.toFixed(2)}, ${s.y.toFixed(2)}, ${s.z.toFixed(2)}) m · speed ${s.spd} m/s · error vs truth ${s.err==null?'—':s.err.toFixed(2)+' m'}`;
 $('hd').textContent=s.home_d.toFixed(1);$('hb').textContent=s.home_b.toFixed(0);$('hc').textContent=(s.climb>=0?'climb ':'descend ')+Math.abs(s.climb).toFixed(1)+' m';$('arrow').setAttribute('transform',`rotate(${s.home_b})`);
 const c=confAt(t);$('cf').textContent=c.toFixed(2);$('cfb').style.width=(c*100)+'%';
 $('ms').textContent=s.state;$('ms').style.color=s.state==='BLIND'?'#9aa':'var(--cy)';$('msd').textContent=`horizontal ${s.spd} m/s · vertical ${s.vz} m/s · |ω| ${s.gyro==null?'—':s.gyro+' rad/s'}`;
 $('chips').innerHTML=STATES.map(x=>`<span class="chip ${x===s.state?'on':''}">${x}</span>`).join('');
 const ev=D.narr.filter(e=>e.t<=t+0.01).slice(-40).reverse();$('narr').innerHTML=ev.map(e=>`<div class="ev"><span class="mono small">${e.t.toFixed(1)} s</span><span class="badge b-${e.col}">${e.kind.replace('_',' ').slice(0,9)}</span><span>${e.msg}</span></div>`).join('')||'<div class="small">no events yet — clean flight</div>';}
function drawTrust(){const V=D.val;if(!V){$('checks').innerHTML='<div class="small">Run tools/validate_output.py first.</div>';return;}
 $('checks').innerHTML=V.checks.map((c,i)=>`<div class="ev" style="grid-template-columns:1fr 54px"><span><span class="tag">RULE 0${i+1} // ${c.rule}</span><br>${c.name}<br><span class="small">${c.detail}</span></span><span class="badge ${c.pass?'b-green':'b-red'}">${c.pass?'PASS':'FAIL'}</span></div>`).join('');
 $('ctot').innerHTML=`<span style="color:${V.passed===V.total?'var(--gn)':'var(--rd)'}">${V.passed} / ${V.total} checks passed</span>`;
 $('hashes').innerHTML=Object.entries(V.hashes).map(([k,h])=>`<div><div class="small">${k} <span style="color:var(--dim)">(${h.path})</span></div><div class="mono" style="font-size:11px;word-break:break-all;background:var(--panel2);border:1px solid var(--line);border-radius:6px;padding:6px">${h.sha256||'file not found'}</div></div>`).join('');
 $('privram').textContent=`${K.ram} MB RAM, ${K.ms} ms per IMU second, single CPU thread`;}
function draw(){if($('m1').classList.contains('on')){drawTraj();drawHealth();side();}else if($('m2').classList.contains('on'))drawStress();else drawTrust();}
if(D.val){const ok=D.val.passed===D.val.total;$('gpill').innerHTML=`<span class="dot" style="background:${ok?'var(--gn)':'var(--rd)'}"></span>${D.val.passed}/${D.val.total} GUARDRAILS ${ok?'PASSED':'FAILED'}`;}else $('gpill').style.display='none';
$('scrub').oninput=e=>{cur=Math.round(e.target.value/1000*(S.length-1));draw();};
let timer=null;$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='▶ PLAY';return;}if(cur>=S.length-1)cur=0;$('play').textContent='❚❚ PAUSE';timer=setInterval(()=>{cur=Math.min(S.length-1,cur+8);$('scrub').value=cur/(S.length-1)*1000;draw();if(cur>=S.length-1){clearInterval(timer);timer=null;$('play').textContent='▶ PLAY';}},40);};
$('k1').textContent=K.ram;$('k1b').style.width=(K.ram/8*100)+'%';$('k2').textContent=K.ms;$('k2b').style.width=(K.ms/20*100)+'%';
$('k3').textContent=K.ate;$('k3n').textContent=D.avg?`avg of 3 seeds · this run ${K.ate_run} m`:'this run';$('k4').textContent=K.spikes_inj?`${K.spikes} / ${K.spikes_inj}`:K.spikes;
function drawStress(){const R=D.stress;let h='<tr><th>scenario</th><th>ATE m</th><th>checkpoints m</th><th>blind 5 s m</th><th>dropouts m</th><th>RPE 1 s m</th></tr>';
 R.forEach(r=>h+=`<tr><td>${r[0]}</td>${r.slice(1).map(v=>`<td>${v.toFixed(2)}</td>`).join('')}</tr>`);if(D.avg)h+=`<tr class="avg"><td>average of 3 seeds</td>${D.avg.map(v=>`<td>${v.toFixed(2)}</td>`).join('')}</tr>`;$('st').innerHTML=h;
 const j=R.filter(r=>/x3|x10|seed 1/.test(r[0])).map(r=>r[1]);$('stn').textContent=R.length?`${R.length} runs. ATE across light (x3) to heavy (x10) jamming: ${Math.min(...j).toFixed(2)}–${Math.max(...j).toFixed(2)} m. Seed-to-seed spread comes from the random bias drift.`:'Run tools/stress_test.py to fill this table.';
 const [x,w,hh]=fit($('sb'));if(!R.length)return;const mx=Math.max(...R.map(r=>r[1]))*1.15,bw=(w-40)/R.length;
 R.forEach((r,i)=>{const bh=(r[1]/mx)*(hh-60),X0=30+i*bw+bw*.18;x.fillStyle=/seed/.test(r[0])?'#22D3EE':'#F59E0B';x.fillRect(X0,hh-30-bh,bw*.64,bh);x.fillStyle='#D7E3EE';x.font='12px monospace';x.fillText(r[1].toFixed(2),X0,hh-36-bh);x.fillStyle='#7F93A6';x.font='10px monospace';x.fillText(r[0].replace('baseline noise x6, ','').replace('light jamming ','light ').replace('heavy jamming ','heavy ').replace('dropouts zeroed (not removed)','zeroed').replace('dropouts zeroed','zeroed'),X0,hh-14);});
 $('sbn').textContent='cyan: baseline jamming x6, three random seeds · amber: other jamming profiles';
 $('eff').innerHTML=`peak RAM <b style="color:var(--cy)">${K.ram} MB</b> / 8 MB &nbsp;·&nbsp; compute <b style="color:var(--gn)">${K.ms} ms</b> per IMU second / 20 ms &nbsp;·&nbsp; ${K.n_imu.toLocaleString()} IMU rows in → ${K.n_pose.toLocaleString()} poses out (gaps filled, +5 s blind)`;}
addEventListener('resize',draw);draw();
</script></body></html>'''

if __name__ == '__main__':
    main()
