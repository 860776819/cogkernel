"""Soul-0 E0 runner: normal / recoverable / beyond-boundary / controls / sweep."""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from proto0 import LAM, K, N, THETA, A_STABLE, RECOVER_LEVEL, simulate

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'soul0_e0')
EQ = np.array([A_STABLE, A_STABLE, N - 2 * A_STABLE])   # sustainable fixed point
T_P = 25.0       # erosion time in the perturbed arms
T_EXT = 800.0    # forward-dynamics horizon proving (non-)re-establishment


def run_arm(name, state0, T, events=(), coupling='both', ext=0.0, dt=0.01):
    ts, traj = simulate(state0, T, dt=dt, events=events, coupling=coupling)
    ext_max = None
    recovered = bool(np.min(traj[-1, :2]) >= RECOVER_LEVEL)
    if ext > 0:
        _, traj2 = simulate(traj[-1], ext, dt=0.05, coupling=coupling)
        ext_max = float(np.max(traj2[:, :2]))
        recovered = recovered and ext_max >= RECOVER_LEVEL
    print(f"{name:28s} end(a,b)=({traj[-1, 0]:.6f}, {traj[-1, 1]:.6f})  "
          f"recovered={recovered}"
          + (f"  max(a,b) over {T_EXT:.0f}-tick horizon = {ext_max:.2e}"
             if ext_max is not None else ""))
    return {'ts': ts, 'traj': traj, 'recovered': recovered, 'ext_max': ext_max,
            'events': events}


def main():
    os.makedirs(OUT, exist_ok=True)
    arms = {
        'normal':       run_arm('E-normal', EQ, 80.0),
        'recover_sym':  run_arm('E-recover(rho=0.5 both)', EQ, 80.0,
                                events=[(T_P, 0.5, 'both')]),
        'recover_asym': run_arm('E-asym(rho=0.9 b)', EQ, 80.0,
                                events=[(T_P, 0.9, 'b')]),
        'beyond':       run_arm('E-beyond(rho=0.99 both)', EQ, 80.0,
                                events=[(T_P, 0.99, 'both')], ext=T_EXT),
        'control_cut':  run_arm('C-cut(relations severed)', EQ, 80.0, coupling=None),
        'control_half': run_arm('C-half(b incorporation off)', EQ, 120.0,
                                coupling='b-only'),
    }

    # sweep: emergent existence boundary, symmetric (both) and asymmetric (b only)
    rhos = np.linspace(0.05, 0.999, 40)
    sweep = []
    for mode in ('both', 'b'):
        for rho in rhos:
            _, traj = simulate(EQ, 150.0, dt=0.02,
                               events=[(25.0, float(rho), mode)])
            rec = bool(np.min(traj[-1, :2]) >= RECOVER_LEVEL)
            sweep.append((mode, float(rho), rec, float(np.min(traj[-1, :2]))))

    def boundary(mode):
        pts = [s for s in sweep if s[0] == mode]
        for i in range(1, len(pts)):
            if pts[i - 1][2] and not pts[i][2]:
                return pts[i][1]
        return None

    rho_star_sym = boundary('both')
    rho_star_asym = boundary('b')
    print(f"sweep: emergent boundary symmetric rho* ~ {rho_star_sym} "
          f"(pre-registered prediction ~0.812); "
          f"asymmetric(b-only) rho* ~ {rho_star_asym}")

    # ---- artifacts ----
    with open(os.path.join(OUT, 'arms_timeseries.csv'), 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['arm', 't', 'a', 'b', 'm'])
        for name, arm in arms.items():
            step = max(1, len(arm['ts']) // 1500)
            for i in range(0, len(arm['ts']), step):
                w.writerow([name, f"{arm['ts'][i]:.3f}",
                            f"{arm['traj'][i, 0]:.6f}", f"{arm['traj'][i, 1]:.6f}",
                            f"{arm['traj'][i, 2]:.6f}"])
    with open(os.path.join(OUT, 'sweep.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['mode', 'rho', 'recovered', 'final_min_ab'])
        for mode, r, rec, v in sweep:
            w.writerow([mode, r, rec, f"{v:.6f}"])

    # ---- figure ----
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))

    def plot_arm(ax, arm, title):
        ts, traj = arm['ts'], arm['traj']
        ax.plot(ts, traj[:, 0], color='tab:blue', label='a')
        ax.plot(ts, traj[:, 1], color='tab:orange', label='b')
        ax.plot(ts, traj[:, 2], color='tab:green', lw=0.8, alpha=0.6, label='m')
        for ev_t, _, _ in arm['events']:
            ax.axvline(ev_t, color='k', lw=0.9, ls='--', alpha=0.7)
        ax.axhline(A_STABLE, color='gray', lw=0.5, ls=':')
        ax.axhline(THETA, color='red', lw=0.5, ls=':')
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel('t')
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc='upper right')

    plot_arm(axs[0, 0], arms['normal'], 'E-normal (sustainable region)')
    plot_arm(axs[0, 1], arms['recover_sym'], 'E-recoverable (rho=0.5 both)')
    plot_arm(axs[0, 2], arms['recover_asym'], 'E-asymmetric (90% of b only)')
    plot_arm(axs[1, 0], arms['beyond'],
             f"E-beyond (rho=0.99): ext horizon max={arms['beyond']['ext_max']:.1e}")
    axC = axs[1, 1]
    tc, tc_ = arms['control_cut']['ts'], arms['control_cut']['traj']
    th, th_ = arms['control_half']['ts'], arms['control_half']['traj']
    axC.plot(tc, tc_[:, 0], color='tab:red', label='C-cut a')
    axC.plot(tc, tc_[:, 1], color='tab:red', ls='--', label='C-cut b')
    axC.plot(th, th_[:, 0], color='tab:purple', label='C-half a')
    axC.plot(th, th_[:, 1], color='tab:purple', ls='--', label='C-half b')
    axC.axhline(THETA, color='gray', lw=0.5, ls=':')
    axC.set_ylim(-0.02, 1.02)
    axC.set_xlabel('t')
    axC.set_title('C-cut (both relations off) / C-half (b off)', fontsize=10)
    axC.legend(fontsize=8)
    rs = [s[1] for s in sweep]
    rec = [s[2] for s in sweep]
    fin = [s[3] for s in sweep]
    mods = [s[0] for s in sweep]
    axS = axs[1, 2]
    for mode, mk in (('both', 'o'), ('b', '^')):
        xs = [r for r, m, c in zip(rs, mods, rec) if m == mode and c]
        ys = [v for v, m, c in zip(fin, mods, rec) if m == mode and c]
        axS.scatter(xs, ys, c='tab:green', s=20, marker=mk,
                    label=f'{mode}: recovered')
        xs = [r for r, m, c in zip(rs, mods, rec) if m == mode and not c]
        ys = [v for v, m, c in zip(fin, mods, rec) if m == mode and not c]
        axS.scatter(xs, ys, c='tab:red', s=20, marker=mk,
                    label=f'{mode}: dissipated')
    for rstar, lbl in [(rho_star_sym, f'ρ*sym≈{rho_star_sym:.2f}'),
                       (rho_star_asym, f'ρ*asym≈{rho_star_asym:.2f}')]:
        if rstar is not None:
            axS.axvline(rstar, color='k', lw=0.8, ls='--', alpha=0.6)
            axS.text(rstar + 0.008, 0.3, lbl, fontsize=8, rotation=90)
    axS.axhline(RECOVER_LEVEL, color='gray', lw=0.5, ls=':')
    axS.set_yscale('log')
    axS.set_xlabel('erosion fraction ρ (both components)')
    axS.set_ylabel('final min(a, b)')
    axS.set_title('emergent existence boundary (sweep)', fontsize=10)
    axS.legend(fontsize=8)

    fig.suptitle('Soul-0 E0 / Proto-0: existence organization a-b-m '
                 '(blue a, orange b, green m; dotted: stable point a2 / presence threshold θ)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'figure_e0.png'), dpi=130)
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
