"""Soul-0 E0.5: internal-discrimination audit of Proto-0 (analysis only).

No changes to the dynamics. Question: does Proto-0 already produce transient
internal differential responses to perturbations that are equal in total
physical magnitude but distributed differently across its own components?

Method: pairs with identical total erosion budget, different distribution
between a and b (conversion-to-m, mass conserving, as in E0). Compare full
transients: a, b, m; regeneration flux p = k*a*b*m; recovery time / dissipation
trend. Plus a budget-by-distribution outcome scan.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from proto0 import K, simulate

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'soul0_e05')
A2 = 0.420782                       # stable fixed point (E0)
N = 1.0
EQ = np.array([A2, A2, N - 2 * A2])
RECOVER_LEVEL = 0.5 * A2
T = 150.0
DT = 0.01


def erode(state, d_a, d_b):
    """Mass-conserving erosion: removed component mass is converted to m."""
    s = state.copy()
    d_a = min(d_a, s[0])
    d_b = min(d_b, s[1])
    s[0] -= d_a
    s[1] -= d_b
    s[2] += d_a + d_b
    return s, (d_a, d_b)


def run_from(state, T=T, dt=DT):
    ts, traj = simulate(state, T, dt=dt)
    p = K * traj[:, 0] * traj[:, 1] * traj[:, 2]
    return ts, traj, p


def recovery_time(ts, traj, level=0.9 * A2):
    ok = np.minimum(traj[:, 0], traj[:, 1]) >= level
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return float('inf')
    i0 = idx[0]
    return float(ts[i0]) if ok[i0:].all() else float('inf')


def metrics(ts, traj, p):
    return {
        'recovered': bool(np.min(traj[-1, :2]) >= RECOVER_LEVEL),
        'min_a': float(np.min(traj[:, 0])),
        'min_b': float(np.min(traj[:, 1])),
        'p_min': float(np.min(p)),
        'p_int': float(np.sum(p) * DT),
        't_rec': recovery_time(ts, traj),
    }


def main():
    os.makedirs(OUT, exist_ok=True)

    # ---- headline pair: budget = a2, symmetric (50/50) vs all-on-b ----
    budget = A2
    s_sym, (da_s, db_s) = erode(EQ, budget / 2, budget / 2)
    s_asym, (da_a, db_a) = erode(EQ, 0.0, budget)
    ts, sym, p_sym = run_from(s_sym)
    _, asym, p_asym = run_from(s_asym)
    m_sym, m_asym = metrics(ts, sym, p_sym), metrics(ts, asym, p_asym)
    print("== headline pair: identical total erosion =", round(budget, 5), "==")
    print(f"  symmetric (da=db={budget/2:.5f}): {m_sym}")
    print(f"  all-on-b (db={budget:.5f}):    {m_asym}")
    div = np.linalg.norm(sym[:, :2] - asym[:, :2], axis=1)
    print(f"  transient divergence ||x_sym - x_asym||: t0={div[0]:.4f} "
          f"peak={div.max():.4f} final={div[-1]:.4f}")

    # ---- fixed-budget distribution sweep: fate vs split fraction ----
    splits = np.linspace(0.0, 1.0, 41)
    dist_scan = []
    for s_frac in splits:
        st, _ = erode(EQ, 0.0, 0.0)
        d_b = budget * float(s_frac)
        d_a = budget - d_b
        d_a = min(d_a, EQ[0]); d_b = min(d_b, EQ[1])
        st = EQ.copy(); st[0] -= d_a; st[1] -= d_b; st[2] += d_a + d_b
        _, traj, _ = run_from(st)
        dist_scan.append((float(s_frac), bool(np.min(traj[-1, :2]) >= RECOVER_LEVEL)))

    # ---- budget scan: same-budget pairs across a range of budgets ----
    budgets = [0.10, 0.20, 0.30, 0.40, 0.42, A2]
    budget_scan = []
    for B in budgets:
        sS, _ = erode(EQ, B / 2, B / 2)
        _, trajS, pS = run_from(sS)
        d_b = min(B, EQ[1])
        d_a = B - d_b
        sA, _ = erode(EQ, d_a, d_b)
        _, trajA, pA = run_from(sA)
        recS = bool(np.min(trajS[-1, :2]) >= RECOVER_LEVEL)
        recA = bool(np.min(trajA[-1, :2]) >= RECOVER_LEVEL)
        d = np.linalg.norm(trajS[:, :2] - trajA[:, :2], axis=1)
        budget_scan.append((B, recS, recA, float(d.max()), recS != recA))
        print(f"budget={B:.3f}: sym recovered={recS}  asym recovered={recA}  "
              f"peak divergence={d.max():.4f}  fate-divergent={recS != recA}")

    # ---- artifacts ----
    with open(os.path.join(OUT, 'pair_timeseries.csv'), 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['t', 'a_sym', 'b_sym', 'a_asym', 'b_asym', 'p_sym', 'p_asym'])
        for i in range(0, len(ts), 5):
            w.writerow([f"{ts[i]:.3f}", f"{sym[i,0]:.6f}", f"{sym[i,1]:.6f}",
                        f"{asym[i,0]:.6f}", f"{asym[i,1]:.6f}",
                        f"{p_sym[i]:.6f}", f"{p_asym[i]:.6f}"])
    with open(os.path.join(OUT, 'distribution_scan.csv'), 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['split_fraction_to_b', 'recovered'])
        for s_frac, rec in dist_scan:
            w.writerow([f"{s_frac:.4f}", rec])
    with open(os.path.join(OUT, 'budget_scan.csv'), 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['budget', 'sym_recovered', 'asym_recovered',
                    'peak_transient_divergence', 'fate_divergent'])
        w.writerows([[b, r1, r2, f"{d:.5f}", fd] for b, r1, r2, d, fd in budget_scan])

    # ---- figure ----
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    ax = axs[0, 0]
    ax.plot(ts, sym[:, 0], color='tab:blue', label='a sym')
    ax.plot(ts, sym[:, 1], color='tab:orange', label='b sym')
    ax.plot(ts, asym[:, 0], color='tab:blue', ls='--', label='a asym')
    ax.plot(ts, asym[:, 1], color='tab:orange', ls='--', label='b asym')
    ax.set_title(f'same total erosion {budget:.3f}: symmetric vs all-on-b')
    ax.set_xlabel('t'); ax.legend(fontsize=8)

    ax = axs[0, 1]
    ax.semilogy(ts, np.maximum(p_sym, 1e-14), color='tab:green', label='p sym')
    ax.semilogy(ts, np.maximum(p_asym, 1e-14), color='tab:red', ls='--', label='p asym')
    ax.set_title('regeneration flux p = k·a·b·m')
    ax.set_xlabel('t'); ax.legend(fontsize=8)

    ax = axs[1, 0]
    xs = [s for s, r in dist_scan]
    ys = [1 if r else 0 for _, r in dist_scan]
    ax.scatter(xs, ys, c=['tab:green' if r else 'tab:red' for _, r in dist_scan], s=25)
    ax.set_xlabel('fraction of budget taken from b (rest from a)')
    ax.set_ylabel('recovered')
    ax.set_yticks([0, 1], ['dissipated', 'recovered'])
    ax.set_title(f'fate vs distribution (fixed budget={budget:.3f})', fontsize=10)

    ax = axs[1, 1]
    for (B, r1, r2, dmax, fd), c in zip(budget_scan,
                                        ['tab:blue', 'tab:cyan', 'tab:purple',
                                         'tab:orange', 'tab:red', 'tab:brown']):
        d_a = min(B / 2, EQ[0]); d_b = min(B - d_a, EQ[1])
        sS, _ = erode(EQ, B / 2, B / 2)
        sA, _ = erode(EQ, d_a, d_b)
        _, trajS, _ = run_from(sS)
        _, trajA, _ = run_from(sA)
        d = np.linalg.norm(trajS[:, :2] - trajA[:, :2], axis=1)
        tsv = np.linspace(0, T, len(d))
        ax.plot(tsv, d, label=f'budget={B:.2f}' + (' FATE-DIVERGENT' if fd else ''))
    ax.set_xlabel('t'); ax.set_ylabel('||x_sym - x_asym||')
    ax.set_title('transient divergence for equal budgets', fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle('Soul-0 E0.5: organization-generated transient differential response '
                 '(same total erosion, different distribution)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'figure_e05.png'), dpi=130)
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
