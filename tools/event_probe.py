"""Internal-event probe for the minus-recall variant (M0M1).

Question (M0_MINUS_DESIGN.md): with every scheduled recall path removed, do
spontaneous, traceable, causally-potent internal events still occur?

Method (all experimenter-side; the brain has no memory->h channel left):
  1. DETECT  wake-only stream (idle excluded: its pulses are scheduled P5 events);
     per tick record maxcos = max_h-key cosine (+ a rotated-h null stream).
     Event = upward crossing of 0.90, refractory 50 ticks (measurement window,
     the brain never sees it).
  2. TRACE   (a) event rate vs rotated-null rate; (b) context lock: distance
     between the current sensation and the passed-by key's sensation at event
     ticks vs the typical pairwise sensation distance in memory.
  3. CAUSE   twin intervention: at each event tick, deepcopy the agent, kick the
     twin's h by a fixed-magnitude random vector, let both run 300 ticks.
     Control: identical kicks at random non-event ticks.
     Causal privilege = event-tick kicks diverge more than random-tick kicks.
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

    rec = {'t': [], 'maxcos': [], 'maxcos_rot': [], 'arg': [], 'x': [], 'count': []}

    def hook(ag):
        n = min(ag.memory.count, ag.memory.cap)
        if n < MIN_KEYS:
            return
        keys = ag.memory.data[:n, 0:32]
        h = ag.kernel.h[:, 0]
        sims = keys @ h / (np.linalg.norm(keys, axis=1) * np.linalg.norm(h) + 1e-9)
        hs = Q @ h
        sims_rot = keys @ hs / (np.linalg.norm(keys, axis=1) * np.linalg.norm(hs) + 1e-9)
        j = int(np.argmax(sims))
        rec['t'].append(ag.tick)
        rec['maxcos'].append(float(sims[j]))
        rec['maxcos_rot'].append(float(sims_rot.max()))
        rec['arg'].append(j)
        rec['x'].append(ag.last_x.copy())
        rec['count'].append(n)

    agent.probe_hook = hook
    for _ in range(WAKE):
        agent.tick_wake(EPS)
    rec = {k: np.array(v) for k, v in rec.items()}
    mem = agent.memory
    events = crossings(rec['maxcos'], THRESH, REFRACTORY, MIN_KEYS)
    events_rot = crossings(rec['maxcos_rot'], THRESH, REFRACTORY, MIN_KEYS)

    # context lock: at event ticks, how close is NOW to the passed-by memory?
    n = min(mem.count, mem.cap)
    senses = mem.data[:n, 36:40]
    ri = np.random.default_rng(seed + 21)
    a = ri.integers(0, n, 2000)
    b = ri.integers(0, n, 2000)
    typical = float(np.linalg.norm(senses[a] - senses[b], axis=1).mean())
    d_event = []
    for i in events:
        pos = np.searchsorted(rec['t'], i)
        if pos < len(rec['t']):
            d_event.append(float(np.linalg.norm(rec['x'][pos] - senses[rec['arg'][pos]])))
    return events, events_rot, (float(np.mean(d_event)) if d_event else float('nan')), typical


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

    event_set = set(event_ticks[:MAX_TWINS])
    ctrl_set = set(control_ticks[:MAX_TWINS])
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
                div_e = abs(main_err[br['t0'] + BRANCH] - br['twin'].err_ema)
                results.append((br['kind'], br['t0'], div_a, div_e))
                done.append(bi)
        for bi in reversed(done):
            branches.pop(bi)
    return results


def main():
    os.makedirs('results', exist_ok=True)
    rows = []
    for seed in SEEDS:
        events, events_rot, d_event, typical = detect_pass(seed)
        print(f"seed{seed}: spontaneous events={len(events)}  "
              f"rotated-null events={len(events_rot)}")
        print(f"         context lock: |x_now-x_key| at events = {d_event:.3f}  "
              f"vs typical pairwise = {typical:.3f}")
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
                de = np.array([x[3] for x in arr])
                print(f"         twin kicks @{kind:8s} n={len(arr):3d}  "
                      f"div_action={da.mean():.4f}±{da.std():.4f}  "
                      f"div_err300={de.mean():.5f}")

    with open('results/event_probe.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['seed', 'kind', 't0', 'div_action_100', 'div_err_300'])
        w.writerows(rows)
    print('-> results/event_probe.csv')


if __name__ == '__main__':
    main()
