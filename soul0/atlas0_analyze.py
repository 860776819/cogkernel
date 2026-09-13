"""Soul-0 Atlas-0 analysis: Boundary-01 flip surface + Gap band statistics.

All FINDINGS numbers must come from the CSV/JSON this script produces.
Outputs (per eps): flip_surface_eps*.csv, gap_bands_eps*.csv, and a combined
atlas0_analysis.json. Observation only.
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from run_atlas0 import OUT, EPSES
from proto1 import K, MU


def flip_surface(eps):
    """Fate-flip R-threshold per (A+B) bin: transition recovered-fraction
    <0.5 -> >=0.5 scanning R upward. Only bins with enough data reported."""
    md = np.load(os.path.join(OUT, f'formal_medium_eps{eps:+.2f}.npz'))
    pts, fate = md['pts'], md['fate']
    S = pts[:, 0] + pts[:, 1]
    R = pts[:, 2]
    sb = np.linspace(0.0, 1.0, 61)
    rb = np.linspace(R.min(), R.max(), 50)
    ii = np.clip(np.digitize(S, sb) - 1, 0, len(sb) - 2)
    ji = np.clip(np.digitize(R, rb) - 1, 0, len(rb) - 2)
    Hrec = np.zeros((len(sb) - 1, len(rb) - 1))
    Hn = np.zeros_like(Hrec)
    for i, j, f in zip(ii, ji, fate):
        Hn[i, j] += 1
        Hrec[i, j] += f
    frac = np.where(Hn >= 20, Hrec / np.maximum(Hn, 1), np.nan)
    rows = []
    for i in range(len(sb) - 1):
        col = frac[i]
        valid = ~np.isnan(col)
        if valid.sum() < 5:
            continue
        above = np.flatnonzero((col >= 0.5) & valid)
        if len(above) == 0:
            rows.append([float(0.5 * (sb[i] + sb[i + 1])), None, None,
                         float(col[valid].min())])
            continue
        t = int(above[0])
        clean = t > 0 and col[t - 1] < 0.5
        rows.append([float(0.5 * (sb[i] + sb[i + 1])),
                     float(0.5 * (rb[t] + rb[t + 1])) if clean else None,
                     bool(clean), float(np.nanmin(col))])
    path = os.path.join(OUT, f'flip_surface_eps{eps:+.2f}.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['A+B bin center', 'flip R (clean transition)', 'clean',
                    'min recovered-fraction in bin-row'])
        w.writerows(rows)
    clean_flips = [(r[0], r[1]) for r in rows if r[2]]
    return rows, clean_flips


def gap_bands(eps, rmax):
    """Reachability shortfall by (A+B, R) band under the CURRENT protocol:
    equilibrium -> single mass-conserving erosion -> free run. 'far' = no
    reachable sample within 0.05. This is protocol-relative, NOT a claim of
    unreachability in any absolute sense."""
    md = np.load(os.path.join(OUT, f'formal_medium_eps{eps:+.2f}.npz'))
    pts, fate = md['pts'], md['fate']
    rd = np.load(os.path.join(OUT, f'reachable_eps{eps:+.2f}.npz'))
    reach, fp = rd['reach'], rd['fp']
    step = max(1, len(reach) // 40000)
    rs = reach[::step]
    far = np.zeros(len(pts), bool)
    for i in range(0, len(pts), 4000):
        ch = pts[i:i + 4000]
        d2 = ((ch[:, None, :] - rs[None, :, :]) ** 2).sum(-1)
        far[i:i + 4000] = d2.min(1) >= 0.05 ** 2
    S = pts[:, 0] + pts[:, 1]
    rows = []
    for mlo, mhi in [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.9), (0.9, 1.01)]:
        for rlo, rhi in [(0.0, 1.0), (1.0, 3.0), (3.0, float(rmax) + 0.1)]:
            sel = (S >= mlo) & (S < mhi) & (pts[:, 2] >= rlo) & (pts[:, 2] < rhi)
            if sel.sum() < 50:
                continue
            rows.append([mlo, mhi, rlo, rhi, int(sel.sum()),
                         float(far[sel].mean()),
                         float(fate[sel].mean())])
    path = os.path.join(OUT, f'gap_bands_eps{eps:+.2f}.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['A+B lo', 'A+B hi', 'R lo', 'R hi', 'n_formal',
                    'far_from_reachable_frac(T=300 protocol)',
                    'recovered_frac'])
        w.writerows(rows)
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {}
    for eps in EPSES:
        rows, clean = flip_surface(eps)
        rmax = 1.05 * (1.0 + max(eps, 0.0)) * K / (27 * MU)
        g = gap_bands(eps, rmax)
        cf = [(round(s, 3), round(r, 3)) for s, r in clean]
        high_far = [r for r in g if r[1] >= 0.1 and r[3] < 3.0 and r[5] >= 0.995]
        print(f"eps={eps:+.2f}: clean flip thresholds (A+B, R) = {cf[:8]}"
              f"{' ...' if len(cf) > 8 else ''}  (n_clean={len(cf)})")
        print(f"  protocol-unvisited bands (A+B>=0.1, R<3, far>=99.5%): "
              f"{len(high_far)}/{sum(1 for r in g if r[1] >= 0.1 and r[3] < 3.0)}")
        out[f'{eps:+.2f}'] = {
            'clean_flips': cf, 'n_flip_bins': len(rows),
            'gap_bands': g, 'n_high_far_bands': len(high_far),
        }
    with open(os.path.join(OUT, 'atlas0_analysis.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print('-> atlas0_analysis.json + flip_surface/gap_bands CSVs')


if __name__ == '__main__':
    main()
