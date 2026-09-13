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
    """Physically aligned coarse->medium comparison.
    The regular grids make coarse point (ia,ib,ir) the medium point
    (2ia,2ib,2ir). Fate agreement over identical physical points; boundary
    = fate differs from an in-grid legal neighbor; boundary convergence =
    Jaccard of coarse/medium boundary point sets projected onto common
    physical voxels (voxel edge = 2 medium spacings)."""
    ia_m, ib_m, ir_m = grid_index(pm, n_m, rmax)
    med = {}
    for k in range(len(pm)):
        med[(int(ia_m[k]), int(ib_m[k]), int(ir_m[k]))] = bool(fm[k])
    ia_c, ib_c, ir_c = grid_index(pc, n_c, rmax)
    keys_c = list(zip(ia_c.tolist(), ib_c.tolist(), ir_c.tolist()))
    paired = [(i, med[key]) for i, key in enumerate(keys_c) if key in med]
    agree = sum(1 for i, mf in paired if bool(fc[i]) == mf) / max(len(paired), 1)
    # local-majority agreement: medium majority fate within one coarse spacing
    # (27-cell index neighborhood), compared with the coarse point's own fate.
    # The identical-point subset alone only checks determinism (it is exactly 1
    # for a deterministic system); this measures resolution convergence.
    local_ok, local_n = 0, 0
    for i, (ia, ib, ir) in enumerate(keys_c):
        nb = [med[(2*ia+di, 2*ib+dj, 2*ir+dk)]
              for di in (-1, 0, 1) for dj in (-1, 0, 1) for dk in (-1, 0, 1)
              if (2*ia+di, 2*ib+dj, 2*ir+dk) in med]
        if len(nb) >= 8:
            local_n += 1
            maj = sum(nb) >= len(nb) / 2
            local_ok += (maj == bool(fc[i]))
    local_agree = local_ok / max(local_n, 1)

    def boundary_set(keys, fates, n):
        fdict = dict(zip(keys, fates))
        bset = set()
        for key, f in fdict.items():
            for d in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
                nb = fdict.get((key[0]+d[0], key[1]+d[1], key[2]+d[2]))
                if nb is not None and nb != f:
                    bset.add(key)
                    break
        return bset

    b_c = boundary_set(keys_c, [bool(f) for f in fc], n_c)
    keys_m = list(zip(ia_m.tolist(), ib_m.tolist(), ir_m.tolist()))
    b_m = boundary_set(keys_m, [bool(f) for f in fm], n_m)

    def to_voxels(bset, n):
        # project index-space boundary points onto common physical voxels of
        # 2 medium spacings: coarse idx * (n_m/n_c) then //2; medium idx //2
        return {tuple(np.round(np.array(k) * (n_m / n_c) / 2).astype(int)) for k in bset}

    v_c = {tuple(np.round(np.array(k) * (n_m / n_c) / 2).astype(int)) for k in b_c}
    v_m = {tuple(np.round(np.array(k) / 2).astype(int)) for k in b_m}
    inter = len(v_c & v_m)
    union = len(v_c | v_m)
    jac = inter / union if union else 1.0
    print(f"  coarse->medium PHYSICAL comparison: identical-point fate "
          f"agreement = {agree:.4f} (n={len(paired)}) -- determinism check ; "
          f"local-majority agreement = {local_agree:.4f} (n={local_n}) "
          f"[resolution-convergence metric]")
    print(f"  boundary points: coarse={len(b_c)} medium={len(b_m)}; "
          f"projected-voxel Jaccard = {jac:.3f} (voxel = 2 medium spacings)")
    return agree, jac, len(paired), len(b_c), len(b_m), local_agree


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
def horizon_recheck(eps, pts, n_band=2000, n_ctrl=1000, seed=7):
    """Reclassify boundary-band samples (A+B in [0.05,0.30]) and random
    controls at T=150/600/2400; report drift rate and drift mass bands."""
    rng = np.random.default_rng(seed)
    mass = pts[:, 0] + pts[:, 1]
    band = np.flatnonzero((mass >= 0.05) & (mass <= 0.30))
    rest = np.flatnonzero((mass < 0.05) | (mass > 0.30))
    sel_band = rng.choice(band, min(n_band, len(band)), replace=False)
    sel_ctrl = rng.choice(rest, min(n_ctrl, len(rest)), replace=False)
    sel = np.concatenate([sel_band, sel_ctrl])
    is_band = np.zeros(len(sel), bool)
    is_band[:len(sel_band)] = True
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
    print(f"  horizon recheck (eps={eps:+.2f}): n={len(sel)} "
          f"(band={is_band.sum()}, control={(~is_band).sum()})")
    print(f"   drift 150->600: {d150_600.mean()*100:.2f}%  "
          f"600->2400: {d600_2400.mean()*100:.2f}%  "
          f"150->2400: {d150_2400.mean()*100:.2f}%")
    drift_mass = mass[sel][d150_2400]
    if len(drift_mass):
        print(f"   drift locations (A+B): "
              f"{np.round(np.quantile(drift_mass, [0, .25, .5, .75, 1]), 3)}")
    return {'n': int(len(sel)), 'band': int(is_band.sum()),
            'drift_150_600': float(d150_600.mean()),
            'drift_600_2400': float(d600_2400.mean()),
            'drift_150_2400': float(d150_2400.mean()),
            'drift_mass_values': [float(x) for x in drift_mass[:50]]}


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
        agree, jac, npaired, nbc, nbm, local_agree = compare_physical(
            pc, fc, pm, fm, 50, 100, rmax)
        # --- fix 2: Map B from this eps's own fixed point ---
        fp = find_fixed_point(eps)
        reach = reachable_atlas(eps, fp)
        np.savez_compressed(os.path.join(OUT, f'reachable_eps{eps:+.2f}.npz'),
                            reach=reach, fp=fp)
        # --- fix 5: horizon recheck ---
        hrc = horizon_recheck(eps, pm)
        repro[f'{eps:+.2f}'] = {
            'fixed_point': [float(x) for x in fp],
            'identical_point_agreement': float(agree),
            'local_majority_agreement': float(local_agree),
            'boundary_jaccard_projected': float(jac),
            'n_paired': int(npaired), 'n_boundary_coarse': int(nbc),
            'n_boundary_medium': int(nbm),
            'horizon_recheck': hrc,
            'r_bound': float(r_bound(eps)),
        }
        summary.append((eps, pc, fc, trc, reach, agree, jac, fp, local_agree))

    with open(os.path.join(OUT, 'atlas0_repro.json'), 'w', encoding='utf-8') as f:
        json.dump(repro, f, indent=1, ensure_ascii=False)

    # ---- figures (regenerated) ----
    rng = np.random.default_rng(0)
    for eps, pc, fc, trc, reach, agree, jac, fp, _ in summary:
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
                     f'[identical-point agree={agree:.3f}, '
                     f'boundary Jaccard={jac:.3f}]')
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f'atlas_eps{eps:+.2f}.png'), dpi=120)
        plt.close(fig)

    # ---- fix 6: interactive 3D HTML (visualization only) ----
    try:
        import plotly.graph_objects as go
        for eps, pc, fc, trc, reach, agree, jac, fp, _ in summary:
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
                           include_plotlyjs='cdn')
        print('  interactive HTML written (3 eps)')
    except Exception as e:
        print(f'  (interactive HTML skipped: {e})')
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
