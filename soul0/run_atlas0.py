"""Soul-0 Atlas-0: state-space atlas of the R1 system (observation only).

Map A (Formal): sample legal (A,B,R) directly, free-run each point, classify
long-term fate. Map B (Reachable): only states produced by legitimate
histories (mass-conserving erosion of the equilibrium, r from its running
value), record what (A,B,R) the trajectories actually visit. The two maps are
never mixed. No noise, no learning, no intervention, no new variables.

Coordinates: (A, B, R), M = 1 - A - B. Legal: A,B >= 0, A+B <= 1, R >= 0.
R sampling bound derived in SOUL0_ATLAS0_DESIGN.md section 1:
  R-hat(eps) = k*G_sup/(27*mu), G_sup = 1+max(eps,0); sample [0, 1.05*R-hat].
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from proto1 import LAM, K, N, A2, MU, R0, EPSES, G, rk4

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'soul0_atlas0')
THETA = 0.05
A2F = 0.420782
R_EQ = LAM * A2F / MU          # equilibrium r ~ 4.2078
T_CLASS = 150.0
DT = 0.02
CHUNK = 4000
RECOVER_LEVEL = 0.5 * A2F
REC_MARK = 0.9 * A2F


def r_bound(eps):
    g_sup = 1.0 + max(eps, 0.0)
    return K * g_sup / (27.0 * MU)


def formal_grid(n, eps):
    """Legal (A,B,R) samples at nominal n per axis. Returns (N,3) array."""
    ab = np.linspace(0.0, 1.0, n + 1)
    rr = np.linspace(0.0, 1.05 * r_bound(eps), n)
    pts = []
    for a in ab:
        for b in ab:
            if a + b <= 1.0 + 1e-12:
                pts.append([(a, b, r) for r in rr])
    return np.array([p for chunk in pts for p in chunk], dtype=float)


def run_chunk(S, T, dt, eps):
    """S: (4, M) state batch. Free-run, return per-column classification:
    fate (True=recovered), t_rec, t_abs, final state (4, M)."""
    n = int(round(T / dt))
    min_ab = np.minimum(S[0], S[1]).copy()
    t_rec = np.full(S.shape[1], np.inf)
    t_abs = np.full(S.shape[1], np.inf)
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


def voxel_summary(pts, fate, n, rmax):
    """Coarse voxelization: (A,B,R) cube -> per-voxel fate stats."""
    vox = {}
    for i, (a, b, r) in enumerate(pts):
        key = (int(a * n), int(b * n), int(r * n / rmax * (n - 1) + 0.5))
        f = vox.setdefault(key, [0, 0])
        f[int(fate[i])] += 1
    return vox


def boundary_voxels(vox):
    """Voxels whose majority fate differs from an adjacent voxel's majority."""
    keys = set(vox.keys())
    bnd = []
    for key, f in vox.items():
        maj = f[1] >= f[0]
        for dx, dy, dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
            nb = vox.get((key[0]+dx, key[1]+dy, key[2]+dz))
            if nb is not None:
                nb_maj = nb[1] >= nb[0]
                if nb_maj != maj:
                    bnd.append(key)
                    break
    return set(bnd)


def reachable_atlas(eps, n_ero=21, T=300.0, dt=0.02, record_every=4):
    """Map B: legitimate histories only. Erosion grid on (d_a, d_b) fractions
    of A2 from the equilibrium; r starts at its running equilibrium value."""
    fr = np.linspace(0.0, 1.0, n_ero)
    inits, meta = [], []
    for fa in fr:
        for fb in fr:
            st = np.array([A2F * (1 - fa), A2F * (1 - fb), 0.0, R_EQ])
            st[2] = 1.0 - st[0] - st[1]
            inits.append(st)
            meta.append((fa, fb))
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


def compare_resolutions(eps, vox_c, vox_m):
    """coarse->medium stability: majority-fate agreement on shared voxels +
    boundary Jaccard."""
    common = set(vox_c.keys()) & set(vox_m.keys())
    if not common:
        return 0.0, 0.0, 0
    agree = 0
    for key in common:
        fc, fm = vox_c[key], vox_m[key]
        if (fc[1] >= fc[0]) == (fm[1] >= fm[0]):
            agree += 1
    bc, bm = boundary_voxels(vox_c), boundary_voxels(vox_m)
    inter = len(bc & bm)
    union = len(bc | bm)
    return agree / len(common), (inter / union if union else 1.0), len(common)


def main():
    os.makedirs(OUT, exist_ok=True)
    summary = []
    for eps in EPSES:
        print(f"===== eps = {eps:+.2f} =====")
        pc, fc, trc, tac, mac, finc = formal_atlas(eps, 50, 'coarse')
        pm, fm, trm, tam, mam, finm = formal_atlas(eps, 100, 'medium')
        rmax_c, rmax_m = r_bound(eps) * 1.05, r_bound(eps) * 1.05
        vox_c = voxel_summary(pc, fc, 50, rmax_c)
        vox_m = voxel_summary(pm, fm, 100, rmax_m)
        agree, jac, ncommon = compare_resolutions(eps, vox_c, vox_m)
        print(f"  coarse->medium: majority-fate agreement={agree:.3f} "
              f"boundary Jaccard={jac:.3f} (shared voxels={ncommon})")
        reach = reachable_atlas(eps)
        # reachable vs formal: formal points never near any reachable sample
        near = np.zeros(len(pc), dtype=bool)
        step = max(1, len(reach) // 40000)
        rs = reach[::step]
        for i in range(0, len(pc), CHUNK):
            ch = pc[i:i+CHUNK]
            d2 = ((ch[:, None, :] - rs[None, :, :]) ** 2).sum(-1)
            near[i:i+CHUNK] = (d2.min(1) < 0.05 ** 2)
        gaps = int((~near).sum())
        print(f"  formal points with no reachable sample within 0.05: {gaps}/{len(pc)} "
              f"({100*gaps/len(pc):.1f}%)  [Gap-01 candidates]")
        # persist
        with open(os.path.join(OUT, f'formal_coarse_eps{eps:+.2f}.csv'), 'w',
                  newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['A', 'B', 'R', 'fate_recovered', 't_rec', 't_abs', 'min_ab'])
            for i in range(len(pc)):
                w.writerow([f"{pc[i,0]:.4f}", f"{pc[i,1]:.4f}", f"{pc[i,2]:.4f}",
                            int(fc[i]), f"{trc[i]:.2f}", f"{tac[i]:.2f}",
                            f"{mac[i]:.5f}"])
        np.savez_compressed(os.path.join(OUT, f'formal_medium_eps{eps:+.2f}.npz'),
                            pts=pm, fate=fm, t_rec=trm, t_abs=tam, min_ab=mam,
                            fin=finm)
        np.savez_compressed(os.path.join(OUT, f'reachable_eps{eps:+.2f}.npz'),
                            reach=reach)
        summary.append((eps, agree, jac, gaps, len(pc), reach))

    # ---- figures: one per eps, 6 panels ----
    for eps, agree, jac, gaps, npts, reach in summary:
        pc, fc = None, None
        # reload coarse from csv (small)
        data = np.genfromtxt(os.path.join(OUT, f'formal_coarse_eps{eps:+.2f}.csv'),
                             delimiter=',', names=True)
        A, B, R = data['A'], data['B'], data['R']
        fate = data['fate_recovered'].astype(bool)
        trec = data['t_rec']
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(2, 3, 1, projection='3d')
        sub = np.random.default_rng(0).choice(len(A), min(6000, len(A)), replace=False)
        ax.scatter(A[sub][fate[sub]], B[sub][fate[sub]], R[sub][fate[sub]],
                   s=2, c='tab:green', label='recovered')
        ax.scatter(A[sub][~fate[sub]], B[sub][~fate[sub]], R[sub][~fate[sub]],
                   s=2, c='tab:red', label='dissipated')
        ax.set_xlabel('A'); ax.set_ylabel('B'); ax.set_zlabel('R')
        ax.set_title(f'Formal atlas 3D (eps={eps:+.2f}, coarse)')
        ax.legend(fontsize=7, loc='upper left')

        # slices at fixed R bands (2 bands) -- panel 2: A-B slice
        ax = fig.add_subplot(2, 3, 2)
        rmid = r_bound(eps) * 1.05 / 2
        sel = np.abs(R - rmid) < r_bound(eps) * 0.05
        ax.scatter(A[sel & fate], B[sel & fate], s=3, c='tab:green', label='rec')
        ax.scatter(A[sel & ~fate], B[sel & ~fate], s=3, c='tab:red', label='diss')
        ax.set_xlabel('A'); ax.set_ylabel('B')
        ax.set_title(f'A-B slice (R ~ {rmid:.2f})')
        ax.legend(fontsize=7)

        # A-R slice (B < 0.02 band)
        ax = fig.add_subplot(2, 3, 3)
        sel = B < 0.02
        ax.scatter(A[sel & fate], R[sel & fate], s=3, c='tab:green')
        ax.scatter(A[sel & ~fate], R[sel & ~fate], s=3, c='tab:red')
        ax.set_xlabel('A'); ax.set_ylabel('R')
        ax.set_title('A-R slice (B ~ 0 edge)')

        # B-R slice (A < 0.02 band)
        ax = fig.add_subplot(2, 3, 4)
        sel = A < 0.02
        ax.scatter(B[sel & fate], R[sel & fate], s=3, c='tab:green')
        ax.scatter(B[sel & ~fate], R[sel & ~fate], s=3, c='tab:red')
        ax.set_xlabel('B'); ax.set_ylabel('R')
        ax.set_title('B-R slice (A ~ 0 edge)')

        # formal vs reachable (A-B projection)
        ax = fig.add_subplot(2, 3, 5)
        ax.scatter(A[::3], B[::3], s=1, c='lightgray', label='formal (all)')
        ax.scatter(reach[:, 0], reach[:, 1], s=1, c='tab:blue', label='reachable')
        ax.set_xlabel('A'); ax.set_ylabel('B')
        ax.set_title('Formal vs Reachable (A-B projection)')
        ax.legend(fontsize=7, markerscale=8)

        # recovery-time gradient (stable layer check)
        ax = fig.add_subplot(2, 3, 6)
        sc = ax.scatter(A, B, c=np.where(np.isfinite(trec), trec, np.nan),
                        cmap='viridis', s=3)
        fig.colorbar(sc, ax=ax, label='t_rec (ticks)')
        ax.set_xlabel('A'); ax.set_ylabel('B')
        ax.set_title('recovery-time layers (formal, coarse)')

        fig.suptitle(f'Soul-0 Atlas-0 / eps={eps:+.2f} '
                     f'(agree={agree:.2f}, boundary Jaccard={jac:.2f}, '
                     f'Gap-01 candidates={gaps}/{npts})')
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f'atlas_eps{eps:+.2f}.png'), dpi=120)
        plt.close(fig)
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
