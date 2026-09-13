"""Soul-0 E1 runner: R1 minimal implementation + falsification battery.

A  eps=0 regression control          (implementation correctness)
B  mirror check                      (a<->b, r unchanged)
C  fading trace                      (two histories -> same (a,b,m), different r)
D  matched-state intervention        (same a,b,m, different r, free-running)
E  frozen-r diagnostic               (clamped-r sections; DIAGNOSTIC ONLY)
F  free-running fate scan            (matched pairs near the sensitive zone)
G  full Jacobian / slow mode         (numeric spectrum at the fixed point)
H  +/- eps contrast                  (structural phenomenon vs feedback sign)
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import proto0
from proto1 import (LAM, K, N, A2, MU, R0, EPSES, G, deriv, rk4,
                    simulate_batch, simulate1, deriv_clamped,
                    simulate_clamped, frozen_r_boundary)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'soul0_e1')
EQ = np.array([A2, A2, N - 2 * A2, LAM * A2 / MU])   # equilibrium guess incl. r
T_SHORT, T_LONG = 150.0, 600.0
DT = 0.01


def erode_state(budget, split):
    s = EQ[:3].copy()          # mass state (a,b,m); callers append r as needed
    d_b = budget * split
    d_a = budget - d_b
    d_a = min(d_a, s[0]); d_b = min(d_b, s[1])
    s[0] -= d_a; s[1] -= d_b; s[2] += d_a + d_b
    return s


def recovered(traj):
    return bool(np.min(traj[-1, :3]) >= 0.5 * A2)


# ---------------- A: eps=0 regression ----------------
def check_A():
    st = erode_state(0.420782, 0.5)
    st4 = np.concatenate([st, [4.2]])
    ts0, tr0 = proto0.simulate(st[:3], 80.0, dt=0.01)
    _, tr1 = simulate_batch(st4, 80.0, 0.01, eps=0.0)
    diff = np.max(np.abs(tr1[:, 0, :3] - tr0))
    mass = np.max(np.abs(tr1[:, 0, :3].sum(1) - 1.0))
    r_range = (tr1[:, 0, 3].min(), tr1[:, 0, 3].max())
    print(f"A eps=0 regression: max|a,b,m diff vs E0| = {diff:.2e}  "
          f"max mass error = {mass:.2e}  r range = [{r_range[0]:.3f},{r_range[1]:.3f}]")
    return diff < 1e-8 and mass < 1e-9


# ---------------- B: mirror check ----------------
def check_B(eps):
    st = erode_state(0.420782, 0.9)          # strongly asymmetric state
    _, t1 = simulate_batch(np.concatenate([st, [4.2]]), 80.0, 0.01, eps=eps)
    _, t2 = simulate_batch(np.concatenate([st[[1, 0, 2]], [4.2]]), 80.0, 0.01, eps=eps)
    d = max(np.max(np.abs(t1[:, 0, 0] - t2[:, 0, 1])),
            np.max(np.abs(t1[:, 0, 1] - t2[:, 0, 0])),
            np.max(np.abs(t1[:, 0, 2] - t2[:, 0, 2])),
            np.max(np.abs(t1[:, 0, 3] - t2[:, 0, 3])))
    print(f"B mirror check (eps={eps:+.2f}): max mirror violation = {d:.2e}")
    return d < 1e-8


# ---------------- C: fading trace ----------------
def check_C(eps):
    h1 = np.concatenate([erode_state(0.420782, 0.5), [4.2]])   # symmetric damage history
    h2 = np.concatenate([erode_state(0.420782, 0.9), [4.2]])   # concentrated damage history
    ts1, tr1 = simulate_batch(h1, 150.0, 0.01, eps=eps)
    _, tr2 = simulate_batch(h2, 150.0, 0.01, eps=eps)
    n = min(len(tr1), len(tr2))
    st_d = np.max(np.abs(tr1[:n, 0, :3] - tr2[:n, 0, :3]), axis=1)
    r_d = np.abs(tr1[:n, 0, 3] - tr2[:n, 0, 3])
    close = np.flatnonzero(st_d < 1e-2)
    if len(close) == 0:
        print(f"C fading trace (eps={eps:+.2f}): states never within 1e-2 -- "
              f"no matched moment found")
        return None
    t_star = close[0]
    print(f"C fading trace (eps={eps:+.2f}): first near-coincidence t={ts1[t_star]:.1f} "
          f"|dstate|={st_d[t_star]:.2e}  |dr|={r_d[t_star]:.4f}  "
          f"|dr| at t={ts1[-1]:.0f}: {r_d[-1]:.4f} (fading)")
    return t_star, float(r_d[t_star]), float(r_d[-1])


# ---------------- D: matched-state intervention ----------------
def check_D(eps, r_values, state):
    ts, trajs = simulate_batch([np.concatenate([state, [r]]) for r in r_values],
                               T_SHORT, 0.01, eps=eps)
    # initial-derivative check (lemma): d(da)/dt difference between extreme clones
    d_pred = K * state[0] * state[1] * state[2] * eps * (
        np.tanh(r_values[0] / R0) - np.tanh(r_values[-1] / R0))
    rate = lambda i: (trajs[1, i, 0] - trajs[0, i, 0]) / 0.01
    d_sim = rate(0) - rate(len(r_values) - 1)
    out = []
    for i in range(len(r_values)):
        for j in range(i + 1, len(r_values)):
            d = np.linalg.norm(trajs[:, i, :3] - trajs[:, j, :3], axis=1)
            sig = np.flatnonzero(d >= 1e-3)
            dur = float(ts[sig[-1]]) if len(sig) else 0.0
            out.append((r_values[i], r_values[j], float(d[1]), dur,
                        recovered_from(trajs[:, i, :3]), recovered_from(trajs[:, j, :3])))
    print(f"D matched-state (eps={eps:+.2f}, state a={state[0]:.3f} b={state[1]:.3f} "
          f"m={state[2]:.3f}): d(da)/dt diff between extreme clones: "
          f"predicted={d_pred:+.5f} simulated={d_sim:+.5f}")
    for r1, r2, d1, dur, f1, f2 in out:
        print(f"   r=({r1:.2f} vs {r2:.2f}): |d(a)/dt| at t0+dt = {d1:.2e} "
              f"divergent until t={dur:.1f}  fates: {f1}/{f2}")
    return out


def recovered_from(traj3):
    return bool(np.min(traj3[-1][:2]) >= 0.5 * A2)   # components only, not m


# ---------------- E: frozen-r diagnostic ----------------
def check_E(eps, r_grid):
    rows = []
    for r in r_grid:
        res = frozen_r_boundary(eps, r)
        if res is None:
            rows.append((r, None, None, None))
            continue
        a1, a2, rho = res
        # numeric spot check: symmetric erosion rho, clamped r
        ok = None
        if r in (0.0, 4.20782, 8.0):
            x = (1.0 - rho * 0.99) * A2          # symmetric state just inside rho
            st4 = np.array([x, x, 1.0 - 2.0 * x, r])
            traj = simulate_clamped(st4, 150.0, 0.01, eps, r)
            ok = bool(np.min(traj[-1, 0, :2]) >= 0.5 * A2)
        rows.append((r, a1, a2, rho, ok))
    for r, a1, a2, rho, ok in rows:
        print(f"E frozen-r (eps={eps:+.2f}) r={r:5.2f}: a1={a1 if a1 is None else round(a1,4)} "
              f"rho*={rho if rho is None else round(rho,4)}"
              + (f"  numeric rho*0.99 recovered={ok}" if ok is not None else ""))
    return rows


# ---------------- F: free-running fate scan (diagnostic-guided) ----------------
def natural_r_range(eps):
    """r range actually produced by legitimate histories (erosion/recovery)."""
    rs = []
    for budget, split in [(0.420782, 0.5), (0.420782, 0.9), (0.420782, 0.0),
                          (0.378522, 0.9), (0.378522, 0.1)]:
        _, tr = simulate_batch(np.concatenate([erode_state(budget, split), [4.2]]),
                               300.0, 0.01, eps=eps)
        rs += [tr[:, 0, 3].min(), tr[:, 0, 3].max()]
    return min(rs), max(rs)


def check_F(eps, r_values, rho_grid):
    """Matched pairs inside the frozen-r diagnostic's disagreement zone.
    States: symmetric erosion rho of EQ -- a=b=(1-rho)*A2, m=1-2a
    (all reachable by valid symmetric erosion). r from natural range."""
    div, rows = [], []
    for rho in rho_grid:
        x = (1.0 - rho) * A2
        st = np.array([x, x, 1.0 - 2.0 * x])
        fates = []
        for r in r_values:
            _, tr = simulate_batch(np.concatenate([st, [r]]), T_SHORT, 0.01, eps=eps)
            fates.append(recovered_from(tr[:, 0, :3]))
        div.append(any(fates) and not all(fates))
        rows.append((float(rho), [float(r) for r in r_values], [bool(f) for f in fates]))
    n_div = sum(div)
    print(f"F free-running scan (eps={eps:+.2f}): {n_div}/{len(rho_grid)} states are "
          f"fate-divergent across r in {[[round(x,2) for x in [r_values[0], r_values[-1]]]]}")
    for rho, rvs, fates in rows:
        if len(set(fates)) > 1:
            print(f"   rho={rho:.3f}: r={rvs} -> fates={fates}  "
                  f"(low-r {'recovers' if fates[0] else 'dissipates'}, "
                  f"high-r {'recovers' if fates[-1] else 'dissipates'})")
    return rows, n_div > 0


# ---------------- F: horizon recheck (no parameter changes) ----------------
def f_horizon_recheck(eps, rho_grid, r_values, horizons=((150.0, 0.01), (600.0, 0.01),
                                                         (2400.0, 0.02))):
    """Re-classify the same rho x r grid at longer horizons. Report whether the
    fate-divergent rho set drifts. Original parameters and r grid unchanged."""
    out = {}
    for T, dt in horizons:
        init, meta = [], []
        for rho in rho_grid:
            x = (1.0 - rho) * A2
            st = np.array([x, x, 1.0 - 2.0 * x])
            for r in r_values:
                init.append(np.concatenate([st, [r]]))
                meta.append((float(rho), float(r)))
        S = np.array(init).T
        n = int(round(T / dt))
        for _ in range(n):
            S = rk4(S, dt, eps)
        fin = S.T
        fates = np.min(fin[:, :2], axis=1) >= 0.5 * A2
        per_rho = {}
        for (rho, r), f in zip(meta, fates):
            per_rho.setdefault(rho, []).append(bool(f))
        div = sorted(rho for rho, fs in per_rho.items() if len(set(fs)) > 1)
        out[T] = div
    stable = all(out[T] == out[150.0] for T, _ in horizons)
    for T, _ in horizons:
        print(f"F horizon recheck (eps={eps:+.2f}) T={T:6.0f}: fate-divergent rho = "
              f"{[round(x, 3) for x in out[T]]}")
    print(f"   stable across horizons: {stable}")
    return out, stable


# ---------------- G: Jacobian spectrum ----------------
ZERO_TOL = 1e-4   # conservation zero-mode: |Re eig| below this = neutral mass mode


def check_G(eps):
    # fixed point: long free run from several starts, collect converged states
    fps = []
    starts = [np.concatenate([EQ[:3], [4.2]]),
              np.concatenate([erode_state(0.420782, 0.5), [1.0]]),
              np.concatenate([erode_state(0.420782, 0.3), [7.0]]),
              np.concatenate([EQ[:3], [1.0]])]
    for s0 in starts:
        ts, tr = simulate_batch(np.asarray(s0), 3000.0, 0.02, eps=eps)
        fp = tr[-1][0, :]
        if np.all(np.abs(deriv(fp[:, None], eps)[:, 0]) < 1e-6) and fp[:3].sum() > 0.5:
            if not any(np.linalg.norm(fp - f) < 1e-3 for f in fps):
                fps.append(fp)
    print(f"G Jacobian (eps={eps:+.2f}): {len(fps)} positive nonzero fixed point(s) "
          f"found within the N=1 conservation manifold (this search)")
    specs = []
    for fp in fps:
        J = np.empty((4, 4))
        for j in range(4):
            e = np.zeros(4); e[j] = 1e-6
            J[:, j] = (deriv(fp[:, None] + e[:, None], eps)[:, 0]
                       - deriv(fp[:, None] - e[:, None], eps)[:, 0]) / 2e-6
        ev, evec = np.linalg.eig(J)
        zero = [z for z in ev if abs(z.real) < ZERO_TOL and abs(z.imag) < ZERO_TOL]
        nonzero = [z for z in ev if not (abs(z.real) < ZERO_TOL and abs(z.imag) < ZERO_TOL)]
        slow = max(nonzero, key=lambda z: z.real)   # stable: closest to 0 from below
        v = np.abs(evec[:, list(ev).index(slow)]); v = v / v.sum()
        stable_manifold = all(z.real < 0 for z in nonzero)
        print(f"   conservation zero-mode(s): {len(zero)} "
              f"(|Re| < {ZERO_TOL}; mass conservation, neutral)")
        print(f"   slow NONZERO mode: Re={slow.real:+.5f}  r-direction "
              f"weight={v[3]:.3f}  (mu={MU})  "
              f"stable within N=1 manifold: {stable_manifold}  "
              f"[full 4D system carries the neutral conservation zero-mode]")
        specs.append((slow.real, float(v[3]), stable_manifold, len(zero)))
    return specs


# ---------------- H orchestration ----------------
def main():
    os.makedirs(OUT, exist_ok=True)
    okA = check_A()
    all_ok_B = all(check_B(e) for e in EPSES)

    # natural r range from a normal + perturbed eps=+0.25 history
    _, tr_nat = simulate_batch(np.concatenate([erode_state(0.420782, 0.5), [4.2]]),
                               300.0, 0.01, eps=0.25)
    r_lo, r_hi = float(tr_nat[:, 0, 3].min()), float(tr_nat[:, 0, 3].max())
    r_values = list(np.linspace(r_lo, r_hi, 5))
    print(f"natural r range (eps=+0.25 history): [{r_lo:.3f}, {r_hi:.3f}] "
          f"-> matched-state r grid = {[round(x,2) for x in r_values]}")

    rows = []
    for eps in EPSES:
        print(f"----- eps = {eps:+.2f} -----")
        if eps != 0.0:
            check_C(eps)
        check_D(eps, r_values, EQ[:3])
        check_D(eps, r_values, erode_state(0.420782, 0.5))
        r_lo, r_hi = natural_r_range(eps)
        r_wide = list(np.linspace(max(r_lo, 0.2), min(r_hi, 10.0), 5))
        print(f"natural r range (eps={eps:+.2f}): [{r_lo:.3f}, {r_hi:.3f}] "
              f"-> scan r grid = {[round(x,2) for x in r_wide]}")
        rho_grid = np.linspace(0.70, 0.90, 41)   # frozen-r diagnostic disagreement zone
        rows_e = check_E(eps, np.linspace(0.0, 10.0, 11))
        fate_rows, fate_div = check_F(eps, r_wide, rho_grid)
        specs = check_G(eps)
        horizon_out, horizon_stable = f_horizon_recheck(eps, rho_grid, r_wide)
        print(f"   [eps={eps:+.2f}] horizon-stable fate bifurcation: {horizon_stable}")
        rows.append((eps, rows_e, fate_div, specs, horizon_out, horizon_stable))
        with open(os.path.join(OUT, f'fate_scan_eps{eps:+.2f}.csv'), 'w', newline='',
                  encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['rho', 'r_grid', 'fates'])
            for rho, rvs, fts in fate_rows:
                w.writerow([f"{rho:.4f}", str(rvs), str(fts)])

    # ---- E: clamped-r rho*(r) curve figure (diagnostic) ----
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    for eps, c in zip(EPSES, ['tab:red', 'tab:gray', 'tab:blue']):
        rs = np.linspace(0.0, 10.0, 60)
        curve = []
        for r in rs:
            res = frozen_r_boundary(eps, r)
            curve.append(res[2] if res else np.nan)
        axs[0, 0].plot(rs, curve, color=c, label=f'eps={eps:+.2f}')
    axs[0, 0].set_title('frozen-r diagnostic: rho*(r) [CLAMPED ONLY]')
    axs[0, 0].set_xlabel('r (clamped)'); axs[0, 0].set_ylabel('rho*')
    axs[0, 0].legend(fontsize=8)

    # fading trace (eps=+0.25)
    h1 = np.concatenate([erode_state(0.420782, 0.5), [4.2]])
    h2 = np.concatenate([erode_state(0.420782, 0.9), [4.2]])
    ts, tr1 = simulate_batch(h1, 150.0, 0.01, eps=0.25)
    _, tr2 = simulate_batch(h2, 150.0, 0.01, eps=0.25)
    n = min(len(tr1), len(tr2))
    axs[0, 1].plot(ts[:n], tr1[:n, 0, 3], label='r history1 (sym)')
    axs[0, 1].plot(ts[:n], tr2[:n, 0, 3], label='r history2 (asym)')
    axs[0, 1].set_title('fading trace: r after different histories')
    axs[0, 1].set_xlabel('t'); axs[0, 1].legend(fontsize=8)

    # matched-state divergence (eps=+0.25, state=post-sym-0.5)
    st = erode_state(0.420782, 0.5)
    _, tm = simulate_batch([np.concatenate([st, [r]]) for r in r_values],
                           T_SHORT, 0.01, eps=0.25)
    for i, r in enumerate(r_values):
        axs[1, 0].plot(ts[:len(tm)], tm[:, i, 0], label=f'a (r={r:.2f})')
    axs[1, 0].set_title('matched states: same a,b,m, different r (free-running)')
    axs[1, 0].set_xlabel('t'); axs[1, 0].legend(fontsize=7)

    # jacobian eigenvalues per eps
    for eps, c in zip(EPSES, ['tab:red', 'tab:gray', 'tab:blue']):
        tsL, trL = simulate_batch(EQ, 3000.0, 0.02, eps=eps)
        fp = trL[-1][0, :]
        J = np.empty((4, 4))
        for j in range(4):
            e = np.zeros(4); e[j] = 1e-6
            J[:, j] = (deriv(fp[:, None] + e[:, None], eps)[:, 0]
                       - deriv(fp[:, None] - e[:, None], eps)[:, 0]) / 2e-6
        ev = np.linalg.eigvals(J)
        axs[1, 1].scatter(ev.real, ev.imag, c=c, label=f'eps={eps:+.2f}', s=30)
    axs[1, 1].axvline(0, color='k', lw=0.5)
    axs[1, 1].set_title('Jacobian eigenvalues at fixed point (per eps)')
    axs[1, 1].set_xlabel('Re'); axs[1, 1].set_ylabel('Im'); axs[1, 1].legend(fontsize=8)

    fig.suptitle('Soul-0 E1 / R1: regulatory freedom (eps pre-registered '
                 '+0.25/0/-0.25; r0=lambda*A2/mu; mu=0.02)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'figure_e1.png'), dpi=130)

    # ---- csv: matched-state summary per eps ----
    with open(os.path.join(OUT, 'matched_states.csv'), 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['eps', 'state', 'r', 'recovered'])
        for eps in EPSES:
            for state, sname in [(EQ[:3], 'EQ'), (erode_state(0.420782, 0.5)[:3], 'mid')]:
                _, trm = simulate_batch([np.concatenate([state, [r]]) for r in r_values],
                                        T_SHORT, 0.01, eps=eps)
                for i, r in enumerate(r_values):
                    w.writerow([eps, sname, r, recovered_from(trm[:, i, :3])])
    print('-> ' + OUT)
    print(f"A regression {'PASS' if okA else 'FAIL'}; B mirror "
          f"{'PASS' if all_ok_B else 'FAIL'}")


if __name__ == '__main__':
    main()
