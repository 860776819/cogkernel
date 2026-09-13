"""Soul-0 Observatory: local offline 3D visualization workbench.

Observation only -- no new mechanisms, no changes to Proto dynamics, no new
experiments, no changes to any Atlas-0 conclusion.

Outputs results/soul0_observatory/index.html (self-contained: plotly.js is
embedded inline, works with no network) + README.txt.

Reachable data is RE-RECORDED with per-trajectory identity (trajectory_id,
fa, fb, t, A, B, R) using the SAME reviewed protocol: each eps's own fixed
point, 21x21 single mass-conserving erosion, T=300, dt=0.02, sample every 4
ticks. An integrity assertion against the existing Atlas-0 reachable npz
guarantees the trajectories are identical.
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from plotly.offline import get_plotlyjs

from proto1 import EPSES, R0
from run_atlas0 import (OUT as ATLAS_OUT, A2F, find_fixed_point,
                        r_bound, R_EQ)

HERE = os.path.dirname(os.path.abspath(__file__))
OBS = os.path.join(HERE, '..', 'results', 'soul0_observatory')
N_ERO = 21
T_TRAJ = 300.0
DT = 0.02
EVERY = 4
FORMAL_REC_SUBSAMPLE = 10000


def trajectories_with_identity(eps, fp):
    """Re-record the reviewed reachable protocol, keeping each trajectory's
    identity. Returns dict(t=[76], ids=[441], fa=[441], fb=[441],
    A/B/R=[[76] x 441])."""
    fr = np.linspace(0.0, 1.0, N_ERO)
    inits = []
    for fa in fr:
        for fb in fr:
            st = np.array([fp[0] * (1 - fa), fp[1] * (1 - fb), 0.0, fp[3]])
            st[2] = 1.0 - st[0] - st[1]
            inits.append(st)
    S = np.array(inits).T
    snaps = [(S[0].copy(), S[1].copy(), S[3].copy())]
    n = int(round(T_TRAJ / DT))
    for i in range(1, n + 1):
        S = rk4_step(S, eps)
        if i % EVERY == 0:
            snaps.append((S[0].copy(), S[1].copy(), S[3].copy()))
    ts = [round(i * EVERY * DT, 2) for i in range(len(snaps))]
    ids, fas, fbs = [], [], []
    A, B, R = [], [], []
    for k in range(N_ERO * N_ERO):
        ids.append(k)
        fas.append(round(float(fr[k // N_ERO]), 4))
        fbs.append(round(float(fr[k % N_ERO]), 4))
        A.append([round(float(snaps[j][0][k]), 4) for j in range(len(snaps))])
        B.append([round(float(snaps[j][1][k]), 4) for j in range(len(snaps))])
        R.append([round(float(snaps[j][2][k]), 4) for j in range(len(snaps))])
    return {'t': ts, 'ids': ids, 'fa': fas, 'fb': fbs, 'A': A, 'B': B, 'R': R}


def rk4_step(S, eps):
    k1 = deriv(S, eps)
    k2 = deriv(S + 0.5 * DT * k1, eps)
    k3 = deriv(S + 0.5 * DT * k2, eps)
    k4 = deriv(S + DT * k3, eps)
    return S + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def deriv(S, eps):
    from proto1 import LAM, K, MU
    a, b, m, r = S[0], S[1], S[2], S[3]
    p = (1.0 + eps * np.tanh(r / R0)) * K * a * b * m
    return np.vstack([-LAM * a + p, -LAM * b + p,
                      LAM * (a + b) - 2 * p, p - MU * r])


def integrity_check(eps, traj):
    """The re-recorded trajectories must match the existing Atlas-0
    reachable npz (same protocol, same order)."""
    rd = np.load(os.path.join(ATLAS_OUT, f'reachable_eps{eps:+.2f}.npz'))
    reach = rd['reach']            # (samples, 3) flattened in (fa, fb) order
    ntraj = N_ERO * N_ERO
    ns = len(traj['t'])
    assert reach.shape[0] == ns * ntraj, (reach.shape, ns * ntraj)
    reach3 = reach.reshape(ns, ntraj, 3)
    for k in (0, 1, 110, 220, 440):
        a0 = reach3[:, k, 0]
        assert np.allclose(a0, traj['A'][k], atol=2e-3), (eps, k, a0[:3],
                                                         traj['A'][k][:3])
        assert np.allclose(reach3[:, k, 1], traj['B'][k], atol=2e-3)
        assert np.allclose(reach3[:, k, 2], traj['R'][k], atol=2e-3)
    print(f"  integrity eps={eps:+.2f}: re-recorded trajectories match "
          f"Atlas-0 reachable npz (checked {ns} samples x 5 ids)")


def load_formal(eps, rng):
    """Coarse formal points for display: all dissipated + subsampled
    recovered. Returns (rec_pts, dis_pts)."""
    path = os.path.join(ATLAS_OUT, f'formal_coarse_eps{eps:+.2f}.csv')
    d = np.genfromtxt(path, delimiter=',', names=True)
    pts = np.vstack([d['A'], d['B'], d['R']]).T
    fate = d['fate_recovered'].astype(bool)
    rec = pts[fate]
    dis = pts[~fate]
    if len(rec) > FORMAL_REC_SUBSAMPLE:
        sel = rng.choice(len(rec), FORMAL_REC_SUBSAMPLE, replace=False)
        rec = rec[sel]
    return rec, dis


def build_html(data_by_eps, fps, rbounds):
    plotly_js = get_plotlyjs()
    assert '</script>' not in plotly_js
    app_js = """
const DATA = __DATA__;
const gd = document.getElementById('gd');
const state = {eps: '+0.00', sel: 220, tIdx: 0, playing: null, camera: null};
const EPS_KEYS = Object.keys(DATA);
const C = {rec: '#2ca02c', dis: '#d62728', line: '#9ecae1', sel: '#ff7f0e',
           now: '#1f77b4', start: '#2b7a0b', end: '#7a1010'};

function rmax(eps) { return DATA[eps].rmax; }

function tracesFor(eps) {
  const D = DATA[eps];
  const tr = [];
  tr.push({type: 'scatter3d', mode: 'markers', name: 'Formal · 持续结局点（抽样）',
    x: D.formal_rec.A, y: D.formal_rec.B, z: D.formal_rec.R,
    marker: {size: 1.6, color: C.rec, opacity: 0.30},
    hoverinfo: 'skip', visible: true});
  tr.push({type: 'scatter3d', mode: 'markers', name: 'Formal · 消散结局点（全部）',
    x: D.formal_dis.A, y: D.formal_dis.B, z: D.formal_dis.R,
    marker: {size: 1.8, color: C.dis, opacity: 0.45},
    hoverinfo: 'skip', visible: true});
  // all-history thin lines as ONE trace with null separators
  const lx = [], ly = [], lz = [], cid = [], ct = [];
  const NT = D.traj.t.length;
  for (let k = 0; k < D.traj.ids.length; k++) {
    for (let i = 0; i < NT; i++) {
      lx.push(D.traj.A[k][i]); ly.push(D.traj.B[k][i]); lz.push(D.traj.R[k][i]);
      cid.push(D.traj.ids[k]); ct.push(D.traj.t[i]);
    }
    lx.push(null); ly.push(null); lz.push(null); cid.push(null); ct.push(null);
  }
  tr.push({type: 'scatter3d', mode: 'lines', name: '全部历史 · 轨迹细线（441 条）',
    x: lx, y: ly, z: lz,
    line: {color: C.line, width: 1},
    opacity: 0.45,
    customdata: cid.map((c, i) => [c, ct[i]]),
    hovertemplate: '历史 #%%{customdata[0]} ｜ t=%%{customdata[1]} ｜ A=%%{x:.3f} ｜ B=%%{y:.3f} ｜ R=%%{z:.3f}<extra></extra>',
    visible: true});
  tr.push({type: 'scatter3d', mode: 'markers', name: '当前时刻 · 全部历史位置',
    x: D.traj.A.map(a => a[0]), y: D.traj.B.map(b => b[0]),
    z: D.traj.R.map(r => r[0]),
    marker: {size: 2.6, color: C.now, opacity: 0.9},
    hovertemplate: '历史 #%%{customdata[0]} ｜ t=%%{customdata[1]} ｜ A=%%{x:.3f} ｜ B=%%{y:.3f} ｜ R=%%{z:.3f}<extra></extra>',
    customdata: D.traj.ids.map((id, k) => [id, D.traj.t[0]]),
    visible: false});
  tr.push({type: 'scatter3d', mode: 'lines', name: '选中历史 · 连续轨迹',
    x: D.traj.A[state.sel], y: D.traj.B[state.sel], z: D.traj.R[state.sel],
    line: {color: C.sel, width: 6},
    customdata: D.traj.t.map(t => [state.sel, t]),
    hovertemplate: '历史 #%%{customdata[0]} ｜ t=%%{customdata[1]} ｜ A=%%{x:.3f} ｜ B=%%{y:.3f} ｜ R=%%{z:.3f}<extra></extra>',
    visible: true});
  tr.push({type: 'scatter3d', mode: 'markers', name: '起点',
    x: [D.traj.A[state.sel][0]], y: [D.traj.B[state.sel][0]],
    z: [D.traj.R[state.sel][0]],
    marker: {size: 5, color: C.start, symbol: 'diamond'},
    hovertemplate: '起点 ｜ 历史 #%{customdata[0]}<extra></extra>',
    customdata: [state.sel], visible: true});
  tr.push({type: 'scatter3d', mode: 'markers', name: '终点',
    x: [D.traj.A[state.sel].at(-1)], y: [D.traj.B[state.sel].at(-1)],
    z: [D.traj.R[state.sel].at(-1)],
    marker: {size: 5, color: C.end, symbol: 'square'},
    hovertemplate: '终点 ｜ 历史 #%{customdata[0]}<extra></extra>',
    customdata: [state.sel], visible: true});
  tr.push({type: 'scatter3d', mode: 'markers', name: '当前时刻 · 选中历史位置',
    x: [D.traj.A[state.sel][state.tIdx]],
    y: [D.traj.B[state.sel][state.tIdx]],
    z: [D.traj.R[state.sel][state.tIdx]],
    marker: {size: 7, color: '#ffbf00', symbol: 'circle',
             line: {color: '#000', width: 1}},
    hovertemplate: '播放头 ｜ 历史 #%{customdata[0]} ｜ t=%{customdata[1]}<extra></extra>',
    customdata: [state.sel, D.traj.t[state.tIdx]], visible: true});
  return tr;
}

function visFromUI() {
  const v = {};
  v['Formal · 持续结局点（抽样）'] = ui.fRec.checked;
  v['Formal · 消散结局点（全部）'] = ui.fDis.checked;
  v['全部历史 · 轨迹细线（441 条）'] = ui.allLines.checked;
  v['当前时刻 · 全部历史位置'] = ui.nowMode.checked;
  v['选中历史 · 连续轨迹'] = ui.single.checked;
  v['起点'] = ui.single.checked && ui.ends.checked;
  v['终点'] = ui.single.checked && ui.ends.checked;
  v['当前时刻 · 选中历史位置'] = ui.single.checked;
  return v;
}

function applyVis() {
  const v = visFromUI();
  const gdTraces = gd.data || [];
  gdTraces.forEach((t, i) => {
    if (t.name in v) Plotly.restyle(gd, {visible: v[t.name]}, [i]);
  });
}

function updateSelection() {
  const D = DATA[state.eps];
  ui.selLabel.textContent = '历史 #' + D.traj.ids[state.sel] +
    '（侵蚀 fa=' + D.traj.fa[state.sel].toFixed(2) +
    '，fb=' + D.traj.fb[state.sel].toFixed(2) + '）';
  Plotly.restyle(gd, {
    x: [D.traj.A[state.sel]], y: [D.traj.B[state.sel]], z: [D.traj.R[state.sel]],
    customdata: [D.traj.t.map(t => [state.sel, t])]}, [4]);
  Plotly.restyle(gd, {x: [[D.traj.A[state.sel][0]]],
    y: [[D.traj.B[state.sel][0]]], z: [[D.traj.R[state.sel][0]]],
    customdata: [[state.sel]]}, [5]);
  Plotly.restyle(gd, {x: [[D.traj.A[state.sel].at(-1)]],
    y: [[D.traj.B[state.sel].at(-1)]], z: [[D.traj.R[state.sel].at(-1)]],
    customdata: [[state.sel]]}, [6]);
  updateMovingDot();
}

function updateMovingDot() {
  const D = DATA[state.eps];
  const i = state.tIdx;
  Plotly.restyle(gd, {x: [[D.traj.A[state.sel][i]]],
    y: [[D.traj.B[state.sel][i]]], z: [[D.traj.R[state.sel][i]]],
    customdata: [[state.sel, D.traj.t[i]]]}, [7]);
  Plotly.restyle(gd, {x: [D.traj.A.map(a => a[i])],
    y: [D.traj.B.map(b => b[i])], z: [D.traj.R.map(r => r[i])],
    customdata: [D.traj.ids.map(id => [id, D.traj.t[i]])]}, [3]);
  ui.tLabel.textContent = 't = ' + D.traj.t[i];
  ui.slider.value = i;
}

function setEps(key) {
  if (state.playing) togglePlay();
  state.eps = key;
  document.querySelectorAll('.epsbtn').forEach(b =>
    b.classList.toggle('active', b.dataset.eps === key));
  const D = DATA[key];
  state.camera = gd.layout && gd.layout.scene && gd.layout.scene.camera
                 ? gd.layout.scene.camera : null;
  const layout = makeLayout(D);
  if (state.camera) layout.scene.camera = state.camera;
  Plotly.react(gd, tracesFor(key), layout);
  fillDropdown(D);
  ui.slider.max = D.traj.t.length - 1;
  state.tIdx = 0;
  updateSelection();
  applyVis();
}

function fillDropdown(D) {
  ui.sel.innerHTML = '';
  D.traj.ids.forEach((id, k) => {
    const o = document.createElement('option');
    o.value = k;
    o.textContent = '历史 #' + id + '（fa=' + D.traj.fa[k].toFixed(2) +
                    ', fb=' + D.traj.fb[k].toFixed(2) + '）';
    if (k === state.sel) o.selected = true;
    ui.sel.appendChild(o);
  });
}

function makeLayout(D) {
  return {
    margin: {l: 0, r: 0, t: 0, b: 0},
    scene: {
      xaxis: {title: 'A', range: [0, 1]},
      yaxis: {title: 'B', range: [0, 1]},
      zaxis: {title: 'R', range: [0, D.rmax]},
      aspectmode: 'data'
    },
    legend: {font: {size: 10}},
    hovermode: 'closest',
    paper_bgcolor: '#fafafa'
  };
}

function togglePlay() {
  if (state.playing) {
    clearInterval(state.playing); state.playing = null;
    ui.play.textContent = '▶ 播放';
  } else {
    state.playing = setInterval(() => {
      const NT = DATA[state.eps].traj.t.length;
      state.tIdx = (state.tIdx + 1) % NT;
      updateMovingDot();
    }, 60);
    ui.play.textContent = '⏸ 暂停';
  }
}

const ui = {};
window.addEventListener('DOMContentLoaded', () => {
  ['fRec', 'fDis', 'allLines', 'nowMode', 'single', 'ends'].forEach(id =>
    ui[id] = document.getElementById(id));
  ui.play = document.getElementById('play');
  ui.slider = document.getElementById('tslider');
  ui.tLabel = document.getElementById('tlabel');
  ui.sel = document.getElementById('tsel');
  ui.selLabel = document.getElementById('sellabel');
  ['fRec', 'fDis', 'allLines', 'nowMode', 'single', 'ends'].forEach(id =>
    ui[id].addEventListener('change', applyVis));
  ui.play.addEventListener('click', togglePlay);
  ui.slider.addEventListener('input', () => {
    state.tIdx = int(ui.slider.value); updateMovingDot();
  });
  ui.sel.addEventListener('change', () => {
    state.sel = int(ui.sel.value);
    ui.single.checked = true;
    applyVis(); updateSelection();
  });
  document.querySelectorAll('.epsbtn').forEach(b =>
    b.addEventListener('click', () => setEps(b.dataset.eps)));
  setEps('+0.00');
});
function int(x) { return parseInt(x, 10); }
"""
    app_js = app_js.replace('__DATA__', json.dumps(data_by_eps,
                                                   separators=(',', ':')))
    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Soul-0 Observatory · 观察工作台</title>
<script>{plotly_js}</script>
<style>
 body {{ margin:0; font-family: 'Microsoft YaHei', sans-serif; display:flex;
        height:100vh; overflow:hidden; background:#eceff1; }}
 #panel {{ width:300px; padding:12px; background:#ffffff; overflow-y:auto;
          box-shadow: 2px 0 6px rgba(0,0,0,.08); z-index:10; }}
 #panel h1 {{ font-size:16px; margin:4px 0 10px; }}
 #panel h2 {{ font-size:12px; color:#555; margin:14px 0 4px;
             border-bottom:1px solid #ddd; padding-bottom:2px; }}
 label {{ display:block; font-size:12.5px; margin:3px 0; cursor:pointer; }}
 .epsbtn {{ display:inline-block; padding:5px 10px; margin:2px; border:1px solid
           #90a4ae; background:#fff; cursor:pointer; font-size:12px;
           border-radius:4px; }}
 .epsbtn.active {{ background:#1f77b4; color:#fff; border-color:#1f77b4; }}
 #gd {{ flex:1; }}
 #play {{ font-size:14px; padding:4px 14px; cursor:pointer; }}
 #tslider {{ width:100%; }}
 #tlabel {{ font-size:12px; color:#333; }}
 #tsel {{ width:100%; font-size:11px; }}
 #sellabel {{ font-size:12px; color:#1f77b4; margin:4px 0; }}
 .note {{ font-size:10.5px; color:#888; margin-top:10px; line-height:1.5; }}
</style>
</head>
<body>
<div id="panel">
 <h1>Soul-0 Observatory · 观察工作台</h1>
 <h2>状态空间 (A, B, R)，M = 1 − A − B</h2>
 <div>
  <span class="epsbtn" data-eps="+0.25">ε = +0.25</span>
  <span class="epsbtn active" data-eps="+0.00">ε = 0</span>
  <span class="epsbtn" data-eps="-0.25">ε = −0.25</span>
 </div>
 <h2>显示图层</h2>
 <label><input type="checkbox" id="fRec" checked> Formal · 持续结局点（抽样 1 万）</label>
 <label><input type="checkbox" id="fDis" checked> Formal · 消散结局点（全部）</label>
 <label><input type="checkbox" id="allLines" checked> 全部历史 · 轨迹细线（441 条）</label>
 <label><input type="checkbox" id="nowMode"> 当前时刻 · 全部历史位置（441 点）</label>
 <h2>单条历史查看</h2>
 <label><input type="checkbox" id="single" checked> 选中历史 · 加亮显示</label>
 <label><input type="checkbox" id="ends" checked> 起点 / 终点标记</label>
 <div id="sellabel"></div>
 <select id="tsel"></select>
 <h2>时间（t = 0 → 300）</h2>
 <button id="play">▶ 播放</button>
 <input type="range" id="tslider" min="0" max="75" value="0" step="1">
 <div id="tlabel">t = 0</div>
 <div class="note">
  坐标：A、B 为组分，R 为调节自由度，M = 1−A−B（守恒）。<br>
  Formal = 数学合法初态直接自由运行；Reachable/历史 = 该 ε 平衡态经单次质量守恒侵蚀
  （21×21 网格，fa/fb 为侵蚀比例）后自由运行 T=300 的真实轨迹。<br>
  颜色仅表示客观分类（持续/消散结局、是否被历史访问），无其他含义。<br>
  悬停任意轨迹点可查看 t / A / B / R / 历史编号。数据与编号来自 Atlas-0 协议；轨迹显示采样为每 ~1.6 t 一点（覆盖全程 t=0→300，完整校验见仓库）。
 </div>
</div>
<div id="gd"></div>
<script>{app_js}</script>
</body>
</html>"""
    return html


def main():
    os.makedirs(OBS, exist_ok=True)
    rng = np.random.default_rng(0)
    data_by_eps, fps, rbounds = {}, {}, {}
    for eps in EPSES:
        print(f"===== eps = {eps:+.2f} =====")
        fp = find_fixed_point(eps)
        traj_full = trajectories_with_identity(eps, fp)
        integrity_check(eps, traj_full)
        # display downsample: 189 points covering t=0..300 (step ~1.6 ticks;
        # dynamics evolve on a ~5-tick scale). Full-resolution integrity is
        # asserted above against the Atlas-0 npz before this.
        sidx = np.unique(np.round(np.linspace(0, len(traj_full['t']) - 1,
                                              189)).astype(int))
        traj = {}
        for key, val in traj_full.items():
            if key in ('ids', 'fa', 'fb'):
                traj[key] = val
            elif key == 't':
                traj[key] = [val[i] for i in sidx]
            else:  # A / B / R: [441][samples]
                traj[key] = [[row[i] for i in sidx] for row in val]
        rec, dis = load_formal(eps, rng)
        rmax = r_bound(eps) * 1.05
        data_by_eps[f'{eps:+.2f}'] = {
            'rmax': round(float(rmax), 4),
            'formal_rec': {'A': [round(float(x), 4) for x in rec[:, 0]],
                           'B': [round(float(x), 4) for x in rec[:, 1]],
                           'R': [round(float(x), 4) for x in rec[:, 2]]},
            'formal_dis': {'A': [round(float(x), 4) for x in dis[:, 0]],
                           'B': [round(float(x), 4) for x in dis[:, 1]],
                           'R': [round(float(x), 4) for x in dis[:, 2]]},
            'traj': traj,
        }
        fps[f'{eps:+.2f}'] = [float(x) for x in fp]
        rbounds[f'{eps:+.2f}'] = float(rmax)
    html = build_html(data_by_eps, fps, rbounds)
    out = os.path.join(OBS, 'index.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    with open(os.path.join(OBS, 'README.txt'), 'w', encoding='utf-8') as f:
        f.write("Soul-0 Observatory（离线观察工作台）\n"
                "==================================\n"
                "双击 index.html 即可打开（无需网络、无需命令行）。\n"
                "数据来源：Atlas-0 协议（各 eps 真实不动点 + 21×21 单次质量守恒侵蚀\n"
                "+ T=300 自由运行），与 results/soul0_atlas0/ 的产物逐点一致。\n")
    print(f"-> {out}  ({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == '__main__':
    main()
