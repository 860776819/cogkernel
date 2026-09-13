"""Internal-event probe for the minus-recall variant (M0M1) -- measurement only.

v2 correction (measurement addendum, 2026-09-13): v1's maxcos over ALL stored
keys is saturated at 1.0 because memory stores an episode every tick, so the
nearest key is always the worm's own immediately-past h. The nearest-neighbor
question is only meaningful over OLD keys (age > AGE_EXCL ticks). v2 reports
both streams; the age-excluded one is primary. The brain is untouched.

  1. DETECT  wake-only stream, maxcos_old = max cosine to keys older than
     AGE_EXCL; rotated-h null over the same old keys (geometric control).
  2. SCAN    thresholds 0.50-0.95: tick-fraction above + upward crossings
     (refractory 50, measurement-side only).
  3. TRACE   context lock for any candidates; causal twins only after
     noise/geometric exclusion (per protocol, next round).
"""
import copy
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np

from agent import Agent

SEEDS = [0, 1]
WAKE = 6000
EPS = 0.15
THRESH = 0.90
REFRACTORY = 50
MIN_KEYS = 300
AGE_EXCL = 300      # a key is 'old' if encoded >= this many ticks ago
KICK = 0.3
BRANCH = 300
DIV_WIN = 100
MAX_TWINS = 30


def crossings(stream, thresh, refractory, min_t):
    out, last = [], -10**9
    for i in range(max(1, min_t), len(stream)):
        if stream[i] >= thresh and stream[i - 1] < thresh and i - last >= refractory:
            out.append(i)
            last = i
    return out


def detect_pass(seed):
    agent = Agent(seed, 'full', 'minus-recall')
    qr = np.random.default_rng(seed + 20)
    Q, _ = np.linalg.qr(qr.normal(0, 1, (32, 32)))

    rec = {'t': [], 'maxcos_all': [], 'maxcos_old': [], 'maxcos_old_rot': [],
           'arg_old': [], 'x': [], 'n_old': []}

    def hook(ag):
        n = min(ag.memory.count, ag.memory.cap)
        if n < MIN_KEYS:
            return
        keys = ag.memory.data[:n, 0:32]
        h = ag.kernel.h[:, 0]
        hn = np.linalg.norm(h) + 1e-9
        kn = np.linalg.norm(keys, axis=1) + 1e-9
        sims = keys @ h / (kn * hn)
        rec['t'].append(ag.tick)
        rec['maxcos_all'].append(float(sims.max()))
        old = ag.memory.t[:n] <= ag.tick - AGE_EXCL
        n_old = int(old.sum())
        rec['n_old'].append(n_old)
        if n_old < MIN_KEYS:
            rec['maxcos_old'].append(np.nan)
            rec['maxcos_old_rot'].append(np.nan)
            rec['arg_old'].append(-1)
            return
        ko = keys[old]
        kon = kn[old]
        j = int(np.argmax(sims[old]))
        hs = Q @ h
        hsn = np.linalg.norm(hs) + 1e-9
        rec['maxcos_old'].append(float(sims[old][j]))
        rec['maxcos_old_rot'].append(float((ko @ hs / (kon * hsn)).max()))
        rec['arg_old'].append(j)
        rec['x'].append(ag.last_x.copy())

    agent.probe_hook = hook
    for _ in range(WAKE):
        agent.tick_wake(EPS)
    rec = {k: np.array(v) for k, v in rec.items()}
    mem = agent.memory
    valid = ~np.isnan(rec['maxcos_old'])
    events = [int(rec['t'][i]) for i in crossings(rec['maxcos_old'], THRESH,
                                                  REFRACTORY, int(valid.argmax()))]
    events_rot = [int(rec['t'][i]) for i in crossings(rec['maxcos_old_rot'], THRESH,
                                                      REFRACTORY, int(valid.argmax()))]

    # context lock: at event ticks, how close is NOW to the passed-by old memory?
    n = min(mem.count, mem.cap)
    senses = mem.data[:n, 36:40]
    ri = np.random.default_rng(seed + 21)
    a = ri.integers(0, n, 2000)
    b = ri.integers(0, n, 2000)
    typical = float(np.linalg.norm(senses[a] - senses[b], axis=1).mean())
    d_event = []
    for i in events:
        pos = np.searchsorted(rec['t'], i)
        if pos < len(rec['t']) and rec['arg_old'][pos] >= 0:
            d_event.append(float(np.linalg.norm(rec['x'][pos] - senses[rec['arg_old'][pos]])))
    return (events, events_rot,
            float(np.mean(d_event)) if d_event else float('nan'), typical, rec)


def twin_pass(seed, event_ticks, control_ticks):
    agent = Agent(seed, 'full', 'minus-recall')
    kr = np.random.default_rng(seed + 22)
    branches = []
    main_acts, main_err = [], []
    results = []

    def spawn(kind, t0):
        twin = copy.deepcopy(agent)
        twin.probe_hook = None
        u = kr.normal(0, 1, 32)
        twin.kernel.h = twin.kernel.h + (KICK * u / np.linalg.norm(u))[:, None]
        branches.append({'kind': kind, 't0': t0, 'twin': twin, 'acts': []})

    event_set = set(t for t in event_ticks[:MAX_TWINS] if t < WAKE - BRANCH - 2)
    ctrl_set = set(t for t in control_ticks[:MAX_TWINS] if t < WAKE - BRANCH - 2)
    print(f"  [twin] event_set={sorted(event_set)} ctrl_set={sorted(ctrl_set)}")
    for t in range(WAKE):
        agent.tick_wake(EPS)
        main_acts.append(agent.cached_a.copy())
        main_err.append(agent.err_ema)
        if t in event_set:
            spawn('event', t)
        if t in ctrl_set:
            spawn('control', t)
        done = []
        for bi, br in enumerate(branches):
            br['twin'].tick_wake(EPS)
            br['acts'].append(br['twin'].cached_a.copy())
            if len(br['acts']) >= BRANCH:
                k = np.array(main_acts[br['t0'] + 1:br['t0'] + 1 + DIV_WIN])
                tw = np.array(br['acts'][:DIV_WIN])
                div_a = float(np.linalg.norm(k - tw, axis=1).mean())
                # twin act k happens at main tick t0+1+k; the 300th act is this tick
                div_e = abs(main_err[t] - br['twin'].err_ema)
                results.append((br['kind'], br['t0'], div_a, div_e))
                done.append(bi)
        for bi in reversed(done):
            branches.pop(bi)
    return results


def main():
    os.makedirs('results', exist_ok=True)
    rows, scan_rows = [], []
    for seed in SEEDS:
        events, events_rot, d_event, typical, rec = detect_pass(seed)
        print(f"seed{seed}: events@0.90(old-key)={len(events)}  "
              f"rotated-null={len(events_rot)}")
        if events:
            print(f"         context lock: |x_now-x_oldkey| at events = {d_event:.3f}"
                  f"  vs typical pairwise = {typical:.3f}")

        with open(f'results/maxcos_stream_seed{seed}.csv', 'w', newline='',
                  encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['tick', 'maxcos_all', 'maxcos_old', 'maxcos_old_rot', 'n_old'])
            for i in range(len(rec['t'])):
                w.writerow([int(rec['t'][i]), f"{rec['maxcos_all'][i]:.5f}",
                            f"{rec['maxcos_old'][i]:.5f}",
                            f"{rec['maxcos_old_rot'][i]:.5f}", int(rec['n_old'][i])])
        for name in ('maxcos_all', 'maxcos_old', 'maxcos_old_rot'):
            v = rec[name][~np.isnan(rec[name])]
            q = np.quantile(v, [0, .05, .25, .5, .75, .95, 1])
            print(f"  {name:15s} quantiles min/5/25/50/75/95/max: "
                  + " ".join(f"{x:.3f}" for x in q))

        print(f"  {'thresh':>6s} {'frac_old':>9s} {'frac_rot':>9s} "
              f"{'ev_old':>7s} {'ev_rot':>7s}   (all-key stream: frac / events)")
        for th in np.arange(0.50, 0.951, 0.05):
            fo = float(np.nanmean(rec['maxcos_old'] >= th))
            frr = float(np.nanmean(rec['maxcos_old_rot'] >= th))
            eo = len(crossings(rec['maxcos_old'], th, REFRACTORY,
                               int((~np.isnan(rec['maxcos_old'])).argmax())))
            ero = len(crossings(rec['maxcos_old_rot'], th, REFRACTORY,
                                int((~np.isnan(rec['maxcos_old_rot'])).argmax())))
            fa = float(np.nanmean(rec['maxcos_all'] >= th))
            ea = len(crossings(rec['maxcos_all'], th, REFRACTORY, MIN_KEYS))
            print(f"  {th:6.2f} {fo:9.4f} {frr:9.4f} {eo:7d} {ero:7d}"
                  f"   {fa:.4f} / {ea}")
            scan_rows.append([seed, round(float(th), 2), fo, frr, eo, ero, fa, ea])

        if events:
            r2 = np.random.default_rng(seed + 23)
            ev = np.array(events[:MAX_TWINS])
            pool = np.arange(400, WAKE - 400)
            ctrl = []
            cand = list(pool)
            r2.shuffle(cand)
            for c in cand:
                if len(ctrl) >= min(len(ev), MAX_TWINS):
                    break
                if all(abs(c - e) > 2 * REFRACTORY for e in events) and \
                   all(abs(c - k) > 2 * REFRACTORY for k in ctrl):
                    ctrl.append(c)
            res = twin_pass(seed, list(events), ctrl)
            for kind, t0, da, de in res:
                rows.append([seed, kind, t0, da, de])
            for kind in ('event', 'control'):
                arr = [r for r in res if r[0] == kind]
                if arr:
                    da = np.array([x[2] for x in arr])
                    print(f"         twin kicks @{kind:8s} n={len(arr):3d}  "
                          f"div_action={da.mean():.4f}±{da.std():.4f}")

    with open('results/event_probe.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['seed', 'kind', 't0', 'div_action_100', 'div_err_300'])
        w.writerows(rows)
    with open('results/threshold_scan.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['seed', 'thresh', 'frac_ticks_old', 'frac_ticks_rot',
                    'events_old', 'events_rot', 'frac_ticks_all', 'events_all'])
        w.writerows(scan_rows)
    print('-> results/event_probe.csv, results/threshold_scan.csv, '
          'results/maxcos_stream_seed*.csv')


if __name__ == '__main__':
    main()
