"""M0-minus driver: Baseline A vs the three subtraction ablations.
Single-factor diffs only (see M0_MINUS_DESIGN.md); arm fixed to 'full'."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import csv

import numpy as np

from experiment import run

VARIANTS = ['baseline-a', 'minus-recall', 'minus-value', 'minus-bridge']
KEYS = ['syn_W1', 'syn_W2', 'syn_W3', 'syn_W4',
        'appr_W1', 'appr_W2', 'appr_W4', 'recovery_W3', 'rel_gain1']


def main(seeds=3):
    print(f"{'variant':14s}" + ''.join(f"{k:>14s}" for k in KEYS))
    rows = []
    for v in VARIANTS:
        runs = [run(s, 'full', 2200, 800, verbose=False, variant=v) for s in range(seeds)]
        cells = []
        for k in KEYS:
            arr = np.array([r[k] for r in runs], dtype=float)
            mu, sd = arr.mean(), arr.std()
            cells.append(f"{mu:8.2f}±{sd:4.2f}")
            rows.append([v, k, mu, sd] + [float(r[k]) for r in runs])
        print(f"{v:14s}" + ''.join(f"{c:>14s}" for c in cells))

    with open('results/m0minus_summary.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['variant', 'metric', 'mean', 'std'] + [f"seed{i}" for i in range(seeds)])
        w.writerows(rows)
    print('-> results/m0minus_summary.csv')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
