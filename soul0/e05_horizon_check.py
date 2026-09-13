"""Horizon-robustness check for the E0.5 dissipation boundary.

The E0.5 bracket (split ~0.00996-0.01006) was measured at T=150 with a
finite-horizon end-state criterion. This script re-runs the endpoint
bisection at longer horizons (T=600, 2400) to test whether the bracket
drifts. Analysis only; proto0.py untouched.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from proto0 import simulate

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'soul0_e05')
A2 = 0.420782
N = 1.0
EQ = np.array([A2, A2, N - 2 * A2])
RECOVER_LEVEL = 0.5 * A2


def recovered_at(split, budget, T, dt):
    d_b = budget * split
    d_a = budget - d_b
    d_a = min(d_a, EQ[0]); d_b = min(d_b, EQ[1])
    st = EQ.copy(); st[0] -= d_a; st[1] -= d_b; st[2] += d_a + d_b
    _, traj = simulate(st, T, dt=dt)
    return bool(np.min(traj[-1, :2]) >= RECOVER_LEVEL)


def bisect(lo, hi, lo_rec, budget, T, dt, tol=1e-4):
    for _ in range(20):
        if hi - lo < tol:
            break
        mid = 0.5 * (lo + hi)
        if recovered_at(mid, budget, T, dt) == lo_rec:
            lo = mid
        else:
            hi = mid
    return lo, hi


def main():
    os.makedirs(OUT, exist_ok=True)
    budget = A2
    rows = []
    for T, dt in [(150.0, 0.01), (600.0, 0.01), (2400.0, 0.02)]:
        lo1, hi1 = bisect(0.0, 0.025, False, budget, T, dt)
        lo2, hi2 = bisect(0.975, 1.0, True, budget, T, dt)
        rows.append((T, lo1, hi1, lo2, hi2))
        print(f"T={T:6.0f}: near-0 boundary [{lo1:.5f}, {hi1:.5f}]  "
              f"near-1 boundary [{lo2:.5f}, {hi2:.5f}]  "
              f"(mirror: 1-hi1={1-hi1:.5f}, 1-lo1={1-lo1:.5f})")
    with open(os.path.join(OUT, 'boundary_horizons.csv'), 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['T', 'near0_lo', 'near0_hi', 'near1_lo', 'near1_hi'])
        w.writerows(rows)
    print('-> results/soul0_e05/boundary_horizons.csv')


if __name__ == '__main__':
    main()
