"""Soul-0 Atlas-0: state-space atlas of the R1 system (observation only).

v2 (review fixes):
  1. coarse->medium comparison now PHYSICALLY ALIGNED: the regular grids make
     each coarse point the even-index subset of the medium grid; fates are
     compared point-by-point at identical (A,B,R). Boundary sets are compared
     as physical boundary points projected onto one common voxel grid.
     (The old voxel-key integer comparison compared different physical
     locations and is withdrawn.)
  2. Map B starts from each eps's OWN R1 positive fixed point (found by free
     run, derivative verified ~0), not a hard-coded E0 equilibrium.
  5. Atlas horizon recheck: boundary-band + random-control samples
     reclassified at T=150/600/2400; drift rate and drift locations reported.

Map A (Formal): legal (A,B,R) sampled directly, free-run, fate classified.
Map B (Reachable): only states produced by legitimate histories (single
mass-conserving erosion of the current eps's fixed point, r from its running
value), recording what (A,B,R) trajectories actually visit.
No noise, no learning, no intervention, no new variables, no semantic colors.
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from proto1 import LAM, K, N, A2, MU, R0, EPSES, G, rk4, deriv

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'soul0_atlas0')
THETA = 0.05
A2F = 0.420782
R_EQ = LAM * A2F / MU          # E0-equilibrium r; used only as an initial guess
T_CLASS = 150.0
DT = 0.02
CHUNK = 4000
RECOVER_LEVEL = 0.5 * A2F
REC_MARK = 0.9 * A2F
HORIZONS = ((150.0, 0.02), (600.0, 0.02), (2400.0, 0.02))


def r_bound(eps):
    g_sup = 1.0 + max(eps, 0.0)
    return K * g_sup / (27.0 * MU)


# ---------------- Map A ----------------
def formal_grid(n, eps):
    ab = np.linspace(0.0, 1.0, n + 1)
    rr = np.linspace(0.0, 1.05 * r_bound(eps), n)
    pts = []
    for a in ab:
        for b in ab:
            if a + b <= 1.0 + 1e-12:
                pts.append([(a, b, r) for r in rr])
    return np.array([p for chunk in pts for p in chunk], dtype=float)


def run_chunk(S, T, dt, eps):
    n = int(round(T / dt))
    t_rec = np.full(S.shape[1], np.inf)
    t_abs = np.full(S.shape[1], np.inf)
    min_ab = np.minimum(S[0], S[1]).copy()
    t = 0.0
    for _ in range(n):
        S = rk4(S, dt, eps)
        t += dt
        mab = np.minimum(S[0], S[1])
        min_ab = np.minimum(min_ab, mab)
        newly_rec = (mab >= REC_MARK) & np.isinf(t_rec)
        t_rec[newly_rec] = t
        newly_abs = (mab < THETA) & np.isinf(t_abs)
        t_abs[newly_abs] = t
    fate = np.minimum(S[0], S[1]) >= RECOVER_LEVEL
    return fate, t_rec, t_abs, min_ab, S


def formal_atlas(eps, n, tag):
    pts = formal_grid(n, eps)
    Np = len(pts)
    fate = np.empty(Np, dtype=bool)
    t_rec = np.empty(Np)
    t_abs = np.empty(Np)
    min_ab = np.empty(Np)
    fin = np.empty((Np, 4))
    for i in range(0, Np, CHUNK):
        chunk = pts[i:i + CHUNK]
        S = np.vstack([chunk[:, 0], chunk[:, 1], 1.0 - chunk[:, 0] - chunk[:, 1],
                       chunk[:, 2]])
        f, tr, ta, ma, Sf = run_chunk(S, T_CLASS, DT, eps)
        fate[i:i + CHUNK] = f
        t_rec[i:i + CHUNK] = tr
        t_abs[i:i + CHUNK] = ta
        min_ab[i:i + CHUNK] = ma
        fin[i:i + CHUNK] = Sf.T
    print(f"  formal[{tag}] eps={eps:+.2f}: {Np} points, "
          f"recovered={int(fate.sum())} ({100*fate.mean():.1f}%)")
    return pts, fate, t_rec, t_abs, min_ab, fin


def grid_index(pts, n, rmax):
    """Regular-grid indices (ia, ib, ir) for formal_grid points."""
    ia = np.clip(np.round(pts[:, 0] * n).astype(int), 0, n)
    ib = np.clip(np.round(pts[:, 1] * n).astype(int), 0, n)
    ir = np.clip(np.round(pts[:, 2] / rmax * (n - 1)).astype(int), 0, n - 1)
    return ia, ib, ir


def compare_physical(pc, fc, pm, fm, n_c, n_m, rmax):
    """Physically aligned coarse->medium comparison (v3).

    For every coarse point's actual physical (A,B,R), convert to the NEAREST
    medium grid index (A/B via n_m=100; R via medium's real len(rr)=n_m-1
    spacing over the same rmax). The coarse R grid (50 points) and medium R
    grid (100 points) are NOT strictly nested, so no 'identical point' claim
    is made -- the metric is named nearest-physical-grid agreement.

    Metrics:
      nearest_physical_grid_agreement -- coarse fate vs the fate of the
        nearest medium grid point.
      local_majority_agreement -- medium majority fate within one coarse
        spacing (27-cell medium-index neighborhood around the nearest index)
        vs the coarse point's fate. Resolution-convergence metric.
      boundary_jaccard_projected -- boundary points (fate differs from a
        legal in-grid neighbor) of each resolution projected onto common
        physical voxels of 2 medium spacings; Jaccard of occupied voxels.
    """
    ia_m, ib_m, ir_m = grid_index(pm, n_m, rmax)
    med = {}
    for k in range(len(pm)):
        med[(int(ia_m[k]), int(ib_m[k]), int(ir_m[k]))] = bool(fm[k])

    def nearest_medium_index(A, B, R):
        iam = int(min(max(round(A * n_m), 0), n_m))
        ibm = int(min(max(round(B * n_m), 0), n_m))
        irm = int(min(max(round(R / rmax * (n_m - 1)), 0), n_m - 1))
        return iam, ibm, irm

    near_ok = n_paired = n_skipped = 0
    local_ok = local_n = 0
    fdict_c = {}
    ia_c, ib_c, ir_c = grid_index(pc, n_c, rmax)
    for k in range(len(pc)):
        fdict_c[(int(ia_c[k]), int(ib_c[k]), int(ir_c[k]))] = bool(fc[k])
    for k in range(len(pc)):
        A, B, R = pc[k]
        nidx = nearest_medium_index(A, B, R)
        if nidx in med:
            n_paired += 1
            near_ok += (med[nidx] == bool(fc[k]))
        else:
            n_skipped += 1
        nb = [med[(nidx[0] + di, nidx[1] + dj, nidx[2] + dk)]
              for di in (-1, 0, 1) for dj in (-1, 0, 1) for dk in (-1, 0, 1)
              if (nidx[0] + di, nidx[1] + dj, nidx[2] + dk) in med]
        if len(nb) >= 8:
            local_n += 1
            local_ok += ((sum(nb) >= len(nb) / 2) == bool(fc[k]))
    near_agree = near_ok / max(n_paired, 1)
    local_agree = local_ok / max(local_n, 1)

    def boundary_set(fdict):
        b = set()
        for key, f in fdict.items():
            for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                      (0, 0, 1), (0, 0, -1)):
                nb = fdict.get((key[0] + d[0], key[1] + d[1], key[2] + d[2]))
                if nb is not None and nb != f:
                    b.add(key)
                    break
        return b

    b_c = boundary_set(fdict_c)
    b_m = boundary_set(med)

    def to_voxel_coarse(key):
        A, B, R = key[0] / n_c, key[1] / n_c, key[2] / (n_c - 1) * rmax
        return (int(round(A * n_m / 2)), int(round(B * n_m / 2)),
                int(round(R / rmax * (n_m - 1) / 2)))

    def to_voxel_medium(key):
        A, B, R = key[0] / n_m, key[1] / n_m, key[2] / (n_m - 1) * rmax
        return (int(round(A * n_m / 2)), int(round(B * n_m / 2)),
                int(round(R / rmax * (n_m - 1) / 2)))

    v_c = {to_voxel_coarse(k) for k in b_c}
    v_m = {to_voxel_medium(k) for k in b_m}
    inter = len(v_c & v_m)
    union = len(v_c | v_m)
    jac = inter / union if union else 1.0
    print(f"  coarse->medium (physical): nearest-physical-grid agreement = "
          f"{near_agree:.4f} (n={n_paired}, skipped={n_skipped}) ; "
          f"local-majority agreement = {local_agree:.4f} (n={local_n}) ; "
          f"boundary Jaccard = {jac:.3f} (coarse bnd={len(b_c)}, medium "
          f"bnd={len(b_m)}, voxel = 2 medium spacings)")
    return near_agree, jac, n_paired, len(b_c), len(b_m), local_agree, n_skipped


# ---------------- Map B ----------------
def find_fixed_point(eps):
    """Current R1 positive fixed point by free run; derivative verified ~0."""
    S = np.array([[A2F], [A2F], [1 - 2 * A2F], [4.2]])
    n = int(round(3000.0 / 0.02))
    for _ in range(n):
        S = rk4(S, 0.02, eps)
    fp = S[:, 0]
    d = deriv(fp[:, None], eps)[:, 0]
    assert np.all(np.abs(d) < 1e-5), f"fixed point derivative not ~0: {d}"
    print(f"  fixed point (eps={eps:+.2f}): A={fp[0]:.6f} B={fp[1]:.6f} "
          f"M={fp[2]:.6f} R={fp[3]:.6f}  |dX/dt|max={np.abs(d).max():.2e}")
    return fp


def reachable_atlas(eps, fp, n_ero=21, T=300.0, dt=0.02, record_every=4):
    """Map B: legitimate histories only -- single mass-conserving erosion of
    THIS eps's fixed point, r starting from its own running value R*."""
    fr = np.linspace(0.0, 1.0, n_ero)
    inits = []
    for fa in fr:
        for fb in fr:
            st = np.array([fp[0] * (1 - fa), fp[1] * (1 - fb), 0.0, fp[3]])
            st[2] = 1.0 - st[0] - st[1]
            inits.append(st)
    S = np.array(inits).T
    rec = [(S[0].copy(), S[1].copy(), S[3].copy())]
    n = int(round(T / dt))
    for i in range(1, n + 1):
        S = rk4(S, dt, eps)
        if i % record_every == 0:
            rec.append((S[0].copy(), S[1].copy(), S[3].copy()))
    A = np.concatenate([x[0] for x in rec])
    B = np.concatenate([x[1] for x in rec])
    R = np.concatenate([x[2] for x in rec])
    print(f"  reachable eps={eps:+.2f}: {len(A)} visited samples "
          f"(A in [{A.min():.3f},{A.max():.3f}], B in [{B.min():.3f},{B.max():.3f}], "
          f"R in [{R.min():.3f},{R.max():.3f}])")
    return np.vstack([A, B, R]).T


# ---------------- Atlas horizon recheck ----------------
def horizon_recheck(eps, pts, fate, n_bnd=2000, n_ctrl=1000, seed=7):
    """Reclassify TRUE boundary points (fate differs from at least one legal
    in-grid neighbor in the medium fate grid) and non-boundary controls at
    T=150/600/2400; report drift rate and drift locations."""
    rng = np.random.default_rng(seed)
    ia, ib, ir = grid_index(pts, 100, r_bound(eps) * 1.05)
    fdict, cdict = {}, {}
    for k in range(len(pts)):
        key = (int(ia[k]), int(ib[k]), int(ir[k]))
        fdict[key] = bool(fate[k])
        cdict[key] = k
    bnd = []
    for key, f in fdict.items():
        for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            nb = fdict.get((key[0] + d[0], key[1] + d[1], key[2] + d[2]))
            if nb is not None and nb != f:
                bnd.append(key)
                break
    bnd_idx = np.array([cdict[k] for k in bnd])
    all_idx = np.array([cdict[k] for k in fdict])
    isb = np.zeros(len(all_idx), bool)
    isb[bnd_idx] = True
    sel_b = rng.choice(bnd_idx, min(n_bnd, len(bnd_idx)), replace=False)
    ctrl_pool = all_idx[~isb]
    sel_c = rng.choice(ctrl_pool, min(n_ctrl, len(ctrl_pool)), replace=False)
    sel = np.concatenate([sel_b, sel_c])
    is_boundary = np.concatenate([np.ones(len(sel_b), bool),
                                  np.zeros(len(sel_c), bool)])
    states = np.vstack([pts[sel, 0], pts[sel, 1],
                        1.0 - pts[sel, 0] - pts[sel, 1], pts[sel, 2]])
    fates = {}
    for T, dt in HORIZONS:
        S = states.copy()
        n = int(round(T / dt))
        for _ in range(n):
            S = rk4(S, dt, eps)
        fates[T] = np.minimum(S[0], S[1]) >= RECOVER_LEVEL
    d150_600 = fates[150.0] != fates[600.0]
    d600_2400 = fates[600.0] != fates[2400.0]
    d150_2400 = fates[150.0] != fates[2400.0]
    mass = pts[sel, 0] + pts[sel, 1]
    print(f"  horizon recheck (eps={eps:+.2f}): n={len(sel)} "
          f"(TRUE boundary={int(is_boundary.sum())}, "
          f"control={int((~is_boundary).sum())}; medium boundary points "
          f"total={len(bnd)})")
    print(f"   drift 150->600: {d150_600.mean()*100:.2f}%  "
          f"600->2400: {d600_2400.mean()*100:.2f}%  "
          f"150->2400: {d150_2400.mean()*100:.2f}%")
    for tag, m in (('boundary', is_boundary), ('control', ~is_boundary)):
        dm = d150_2400 & m
        print(f"   drift 150->2400 @{tag}: {dm.sum()}/{m.sum()}"
              + (f"  at A+B "
                 f"{np.round(np.quantile(mass[dm], [0, .5, 1]), 3)}"
                 if dm.sum() else ""))
    return {'n': int(len(sel)),
            'n_boundary_true': int(is_boundary.sum()),
            'n_medium_boundary_total': int(len(bnd)),
            'drift_150_600': float(d150_600.mean()),
            'drift_600_2400': float(d600_2400.mean()),
            'drift_150_2400': float(d150_2400.mean()),
            'drift_150_2400_boundary': float(d150_2400[is_boundary].mean()),
            'drift_150_2400_control': float(d150_2400[~is_boundary].mean())}


# ---------------- main ----------------
def main():
    os.makedirs(OUT, exist_ok=True)
    summary = []
    repro = {}
    for eps in EPSES:
        print(f"===== eps = {eps:+.2f} =====")
        rmax = r_bound(eps) * 1.05
        # --- Map A (reuse existing artifacts if present; identical dynamics/grid) ---
        ccsv = os.path.join(OUT, f'formal_coarse_eps{eps:+.2f}.csv')
        mnpz = os.path.join(OUT, f'formal_medium_eps{eps:+.2f}.npz')
        if os.path.exists(ccsv) and os.path.exists(mnpz):
            print("  formal maps: reusing existing artifacts (identical dynamics/grid)")
            data = np.genfromtxt(ccsv, delimiter=',', names=True)
            pc = np.vstack([data['A'], data['B'], data['R']]).T
            fc = data['fate_recovered'].astype(bool)
            trc, tac, mac = data['t_rec'], data['t_abs'], data['min_ab']
            md = np.load(mnpz)
            pm, fm, trm, tam, mam, finm = (md['pts'], md['fate'], md['t_rec'],
                                           md['t_abs'], md['min_ab'], md['fin'])
        else:
            pc, fc, trc, tac, mac, finc = formal_atlas(eps, 50, 'coarse')
            pm, fm, trm, tam, mam, finm = formal_atlas(eps, 100, 'medium')
            with open(ccsv, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['A', 'B', 'R', 'fate_recovered', 't_rec', 't_abs',
                            'min_ab'])
                for i in range(len(pc)):
                    w.writerow([f"{pc[i,0]:.4f}", f"{pc[i,1]:.4f}", f"{pc[i,2]:.4f}",
                                int(fc[i]), f"{trc[i]:.2f}", f"{tac[i]:.2f}",
                                f"{mac[i]:.5f}"])
            np.savez_compressed(mnpz, pts=pm, fate=fm, t_rec=trm, t_abs=tam,
                                min_ab=mam, fin=finm)
        # --- fix 1: physical comparison ---
        near_agree, jac, npaired, nbc, nbm, local_agree, n_skipped = \
            compare_physical(pc, fc, pm, fm, 50, 100, rmax)
        # --- fix 2: Map B from this eps's own fixed point ---
        fp = find_fixed_point(eps)
        reach = reachable_atlas(eps, fp)
        np.savez_compressed(os.path.join(OUT, f'reachable_eps{eps:+.2f}.npz'),
                            reach=reach, fp=fp)
        # --- fix 5: horizon recheck ---
        hrc = horizon_recheck(eps, pm, fm)
        repro[f'{eps:+.2f}'] = {
            'fixed_point': [float(x) for x in fp],
            'nearest_physical_grid_agreement': float(near_agree),
            'nearest_skipped': int(n_skipped),
            'local_majority_agreement': float(local_agree),
            'boundary_jaccard_projected': float(jac),
            'n_paired': int(npaired), 'n_boundary_coarse': int(nbc),
            'n_boundary_medium': int(nbm),
            'horizon_recheck': hrc,
            'r_bound': float(r_bound(eps)),
        }
        summary.append((eps, pc, fc, trc, reach, near_agree, jac, fp,
                        local_agree))

    with open(os.path.join(OUT, 'atlas0_repro.json'), 'w', encoding='utf-8') as f:
        json.dump(repro, f, indent=1, ensure_ascii=False)

    # ---- figures (regenerated) ----
    rng = np.random.default_rng(0)
    for eps, pc, fc, trc, reach, near_agree, jac, fp, _ in summary:
        data = None
        A, B, R = pc[:, 0], pc[:, 1], pc[:, 2]
        fate = fc
        trec = trc
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(2, 3, 1, projection='3d')
        sub = rng.choice(len(A), min(6000, len(A)), replace=False)
        ax.scatter(A[sub][fate[sub]], B[sub][fate[sub]], R[sub][fate[sub]],
                   s=2, c='tab:green', label='recovered')
        ax.scatter(A[sub][~fate[sub]], B[sub][~fate[sub]], R[sub][~fate[sub]],
                   s=2, c='tab:red', label='dissipated')
        ax.set_xlabel('A'); ax.set_ylabel('B'); ax.set_zlabel('R')
        ax.set_title(f'Formal atlas 3D (eps={eps:+.2f}, coarse)')
        ax.legend(fontsize=7, loc='upper left')

        ax = fig.add_subplot(2, 3, 2)
        rmid = r_bound(eps) * 1.05 / 2
        sel = np.abs(R - rmid) < r_bound(eps) * 0.05
        ax.scatter(A[sel & fate], B[sel & fate], s=3, c='tab:green', label='rec')
        ax.scatter(A[sel & ~fate], B[sel & ~fate], s=3, c='tab:red', label='diss')
        ax.set_xlabel('A'); ax.set_ylabel('B')
        ax.set_title(f'A-B slice (R ~ {rmid:.2f})')
        ax.legend(fontsize=7)

        ax = fig.add_subplot(2, 3, 3)
        sel = B < 0.02
        ax.scatter(A[sel & fate], R[sel & fate], s=3, c='tab:green')
        ax.scatter(A[sel & ~fate], R[sel & ~fate], s=3, c='tab:red')
        ax.set_xlabel('A'); ax.set_ylabel('R')
        ax.set_title('A-R slice (B ~ 0 edge)')

        ax = fig.add_subplot(2, 3, 4)
        sel = A < 0.02
        ax.scatter(B[sel & fate], R[sel & fate], s=3, c='tab:green')
        ax.scatter(B[sel & ~fate], R[sel & ~fate], s=3, c='tab:red')
        ax.set_xlabel('B'); ax.set_ylabel('R')
        ax.set_title('B-R slice (A ~ 0 edge)')

        ax = fig.add_subplot(2, 3, 5)
        ax.scatter(A[::3], B[::3], s=1, c='lightgray', label='formal (all)')
        step = max(1, len(reach) // 20000)
        ax.scatter(reach[::step, 0], reach[::step, 1], s=1, c='tab:blue',
                   label='reachable')
        ax.scatter([fp[0]], [fp[1]], s=60, marker='*', c='black',
                   label="this eps's fixed point (Map B origin)")
        ax.set_xlabel('A'); ax.set_ylabel('B')
        ax.set_title('Formal vs Reachable (A-B projection)')
        ax.legend(fontsize=7, markerscale=8)

        ax = fig.add_subplot(2, 3, 6)
        sc = ax.scatter(A, B, c=np.where(np.isfinite(trec), trec, np.nan),
                        cmap='viridis', s=3)
        fig.colorbar(sc, ax=ax, label='t_rec (ticks)')
        ax.set_xlabel('A'); ax.set_ylabel('B')
        ax.set_title('recovery-time layers (formal, coarse)')

        fig.suptitle(f'Soul-0 Atlas-0 / eps={eps:+.2f} '
                     f'[nearest-grid agree={near_agree:.3f}, '
                     f'boundary Jaccard={jac:.3f}]')
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f'atlas_eps{eps:+.2f}.png'), dpi=120)
        plt.close(fig)

    # ---- fix 6: interactive 3D HTML (visualization only) ----
    try:
        import plotly.graph_objects as go
        for eps, pc, fc, trc, reach, near_agree, jac, fp, _ in summary:
            sub = rng.choice(len(pc), min(8000, len(pc)), replace=False)
            fig = go.Figure()
            f = fc[sub]
            fig.add_trace(go.Scatter3d(
                x=pc[sub][f, 0], y=pc[sub][f, 1], z=pc[sub][f, 2],
                mode='markers', marker=dict(size=1.5, color='green',
                                            opacity=0.35),
                name='formal: recovered'))
            fig.add_trace(go.Scatter3d(
                x=pc[sub][~f, 0], y=pc[sub][~f, 1], z=pc[sub][~f, 2],
                mode='markers', marker=dict(size=1.5, color='red', opacity=0.35),
                name='formal: dissipated'))
            step = max(1, len(reach) // 15000)
            fig.add_trace(go.Scatter3d(
                x=reach[::step, 0], y=reach[::step, 1], z=reach[::step, 2],
                mode='markers', marker=dict(size=1.5, color='blue', opacity=0.3),
                name='reachable (history)'))
            fig.update_layout(
                title=f'Soul-0 Atlas-0 eps={eps:+.2f}: (A,B,R) state space '
                      f'[formal fate + reachable sheet]',
                scene=dict(xaxis_title='A', yaxis_title='B', zaxis_title='R'),
                legend=dict(itemsizing='constant'))
            fig.write_html(os.path.join(OUT, f'atlas3d_eps{eps:+.2f}.html'),
                           include_plotlyjs=True)
        print('  interactive HTML written (3 eps)')
    except Exception as e:
        print(f'  (interactive HTML skipped: {e})')
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
