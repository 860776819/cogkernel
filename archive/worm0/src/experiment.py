"""E0: offline-thinking experiment (plan section 5).

Protocol per seed:
  W1(wake) -> I1(frozen) -> W2(wake, same world) -> [world rerolled unseen]
  -> W3(wake) -> I2(frozen) -> W4(wake)
Arms: full (LP-guided replay + bandit), uniform (same replay compute, uniform
cluster choice), none (kernel drift only, no internal work).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import csv

import numpy as np

from agent import Agent

FULL = [('W1', 2200, 'wake', False),
        ('I1', 1200, 'idle', False),
        ('W2', 2200, 'wake', False),
        ('W3', 2200, 'wake', True),
        ('I2', 1200, 'idle', False),
        ('W4', 2200, 'wake', False)]
ARMS = ['full', 'uniform', 'none']


def entropy_bits(counts):
    p = counts / max(counts.sum(), 1)
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())


def run(seed, arm, lengths, windows, verbose=True, variant='baseline-a'):
    agent = Agent(seed, arm, variant)
    fac = lengths / 2200.0
    phases = [(n, int(round(l * fac)), k, rr) for (n, l, k, rr) in FULL]

    tick_rows = []          # (phase_idx, err_ema, C, energy, approach, syncopes)
    phase_err = {}          # name -> list of err_ema
    phase_raw = {}          # name -> list of raw per-tick wake errors
    phase_syn = {}
    phase_appr = {}

    for pi, (name, length, kind, reroll) in enumerate(phases):
        if reroll:
            agent.world.reset_fields()
        errs, raws, appr = [], [], []
        syn_start = agent.syncopes
        for t in range(length):
            if kind == 'wake':
                eps = _eps(name, t, length)
                err = agent.tick_wake(eps)
                errs.append(agent.err_ema)
                raws.append(err)
                appr.append(agent.approach_ema)
            else:
                agent.tick_idle()
                errs.append(agent.err_ema)
            tick_rows.append((pi, agent.err_ema, agent.C, agent.world.energy,
                              agent.approach_ema, agent.syncopes))
        phase_err[name] = np.array(errs)
        phase_raw[name] = np.array(raws)
        phase_appr[name] = float(np.mean(appr)) if appr else 0.0
        phase_syn[name] = agent.syncopes - syn_start
        if kind == 'wake':
            # drop delayed dC labels that would straddle a phase boundary:
            # idle-phase C changes must not be attributed to wake-time states
            agent.dc_queue.clear()
        if verbose:
            print(f"  seed{seed} {arm:8s} {name}: err_ema={errs[-1]:.4f} "
                  f"C={agent.C:.3f} energy={agent.world.energy:.2f} "
                  f"syncope+={phase_syn[name]} appr={phase_appr[name]:+.3f} "
                  f"pulses={agent.pulses}")

    w = windows
    def seg(ph, sl):
        return float(np.mean(phase_err[ph][sl]))

    gains = {
        'gain1': seg('W1', slice(-w, None)) - seg('W2', slice(0, w)),
        'gain2': seg('W3', slice(-w, None)) - seg('W4', slice(0, w)),
        'e_w1_end': seg('W1', slice(-w, None)),
        'e_w2_start': seg('W2', slice(0, w)),
        'e_w3_end': seg('W3', slice(-w, None)),
    }
    gains['rel_gain1'] = gains['gain1'] / (gains['e_w1_end'] + 1e-9)
    gains['rel_gain2'] = gains['gain2'] / (gains['e_w3_end'] + 1e-9)

    # recovery after the unseen world change: measured on raw error (EMA lags)
    w2r, w3r = phase_raw['W2'], phase_raw['W3']
    err_pre = float(np.mean(w2r[-300:]))
    k = 25
    smooth = np.convolve(w3r, np.ones(k) / k, mode='valid')
    peak = int(np.argmax(smooth))
    if smooth[peak] <= 1.2 * err_pre:
        recovery_w3 = 0  # no measurable disruption
    else:
        hit = np.flatnonzero(smooth[peak:] <= 1.1 * err_pre)
        recovery_w3 = int(hit[0]) if len(hit) else len(smooth) - peak

    cb, cd, _ = agent.cos_s4()
    ent_ext_idle = sum(entropy_bits(agent.ent['idle'][d]) for d in range(2))
    ent_ext_wake = sum(entropy_bits(agent.ent['wake'][d]) for d in range(2))
    ent_int_idle = sum(entropy_bits(agent.ent['idle'][d]) for d in range(2, 4))

    return {
        'seed': seed, 'arm': arm, 'variant': variant,
        'e_w1_end': gains['e_w1_end'], 'e_w2_start': gains['e_w2_start'],
        'rel_gain1': gains['rel_gain1'], 'rel_gain2': gains['rel_gain2'],
        'syn_W1': phase_syn['W1'], 'syn_W2': phase_syn['W2'],
        'syn_W3': phase_syn['W3'], 'syn_W4': phase_syn['W4'],
        'recovery_W3': recovery_w3,
        'appr_W1': phase_appr['W1'], 'appr_W2': phase_appr['W2'],
        'appr_W3': phase_appr['W3'], 'appr_W4': phase_appr['W4'],
        'cos_before': cb if cb is not None else 0.0,
        'cos_during': cd if cd is not None else 0.0,
        'ent_ext_idle': ent_ext_idle, 'ent_ext_wake': ent_ext_wake,
        'ent_int_idle': ent_int_idle,
        'tick_rows': tick_rows,
        'bandit_hist': list(agent.bandit.history),
    }


def _eps(name, t, length):
    if name == 'W1':
        return max(0.15, 1.0 - 1.1 * t / length)
    if name == 'W3':
        return 0.25
    return 0.15


def mean_std(vals):
    v = np.array(vals, dtype=float)
    return float(v.mean()), float(v.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--seeds', type=int, default=3)
    args = ap.parse_args()

    lengths = 440 if args.quick else 2200
    windows = 160 if args.quick else 800
    fac = lengths / 2200.0

    os.makedirs('results', exist_ok=True)
    runs = {}
    for arm in ARMS:
        runs[arm] = [run(s, arm, lengths, windows) for s in range(args.seeds)]

    # ---- aggregate + verdicts ----
    def m(arm, key):
        return mean_std([r[key] for r in runs[arm]])

    lines = []
    def say(s=''):
        print(s)
        lines.append(s)

    say('=' * 72)
    say(f"E0 offline-thinking experiment   ({'QUICK' if args.quick else 'FULL'}, "
        f"{args.seeds} seeds/arm)")
    say('=' * 72)
    say(f"{'metric':28s}{'full':>14s}{'uniform':>14s}{'none':>14s}")
    for key in ['rel_gain1', 'rel_gain2', 'syn_W1', 'syn_W2', 'syn_W3', 'syn_W4',
                'recovery_W3', 'appr_W1', 'appr_W2', 'appr_W3', 'appr_W4',
                'ent_ext_idle', 'ent_ext_wake', 'cos_before', 'cos_during']:
        cells = []
        for arm in ARMS:
            mu, sd = m(arm, key)
            cells.append(f"{mu:8.3f}±{sd:4.2f}")
        say(f"{key:28s}" + ''.join(f"{c:>14s}" for c in cells))

    g1f, g1f_sd = m('full', 'rel_gain1'); g1n, _ = m('none', 'rel_gain1'); g1u, _ = m('uniform', 'rel_gain1')
    g2f, _ = m('full', 'rel_gain2'); g2n, _ = m('none', 'rel_gain2'); g2u, _ = m('uniform', 'rel_gain2')
    sw1f, _ = m('full', 'syn_W1'); sw2f, _ = m('full', 'syn_W2')
    sw1n, _ = m('none', 'syn_W1'); sw2n, _ = m('none', 'syn_W2')
    rec_f, _ = m('full', 'recovery_W3'); rec_n, _ = m('none', 'recovery_W3')
    cbf, _ = m('full', 'cos_before'); cdf, _ = m('full', 'cos_during')

    say('-' * 72)

    def verdict(name, ok, weak, detail):
        tag = 'PASS' if ok else ('WEAK' if weak else 'FAIL')
        say(f"[{tag:4s}] {name:34s} {detail}")
        return tag

    verdict('S1  offline gain (full vs none)',
            (g1f - g1n >= 0.05) and (g1f >= 0.08),
            (g1f - g1n >= 0.02) and (g1f >= 0.05),
            f"rel_gain1 full={g1f:.3f} uniform={g1u:.3f} none={g1n:.3f}")
    verdict("S1' selection value (full vs uniform)",
            g1f - g1u >= 0.03, g1f - g1u >= 0.015,
            f"diff={g1f - g1u:.3f}")
    red_f, red_n = sw1f - sw2f, sw1n - sw2n
    verdict('S2  behavior change (syncope drop)',
            red_f - red_n >= 0.5 and red_f >= 1.0,
            red_f - red_n >= 0.25 and red_f >= 0.5,
            f"reduction full={red_f:.2f} none={red_n:.2f}")
    verdict('S3  strategy transfer (W3 recovery)',
            rec_f <= 0.85 * rec_n, rec_f <= 0.95 * rec_n,
            f"recovery_W3 full={rec_f:.0f} none={rec_n:.0f}")
    cosd = cdf - cbf
    verdict('S4  structured idle (pulse cos)',
            cosd >= 0.05, cosd >= 0.02,
            f"cos during={cdf:.3f} before={cbf:.3f} delta={cosd:.3f}")
    say('-' * 72)
    say(f"S0  info audit: EXTEROceptive entropy bits/tick  wake={m('full','ent_ext_wake')[0]:.2f}  "
        f"idle={m('full','ent_ext_idle')[0]:.2f}  (idle~0 means gains cannot come from new info; "
        f"intero idle={m('full','ent_int_idle')[0]:.2f})")

    with open('results/verdicts.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    # ---- csv ----
    with open('results/summary.csv', 'w', newline='', encoding='utf-8') as f:
        wcsv = csv.writer(f)
        keys = ['seed', 'arm', 'e_w1_end', 'e_w2_start', 'rel_gain1', 'rel_gain2',
                'syn_W1', 'syn_W2', 'syn_W3', 'syn_W4', 'recovery_W3',
                'appr_W1', 'appr_W2', 'appr_W3', 'appr_W4',
                'cos_before', 'cos_during', 'ent_ext_idle', 'ent_ext_wake', 'ent_int_idle']
        wcsv.writerow(keys)
        for arm in ARMS:
            for r in runs[arm]:
                wcsv.writerow([r[k] for k in keys])

    # seed-averaged downsampled tick curves per arm
    ds = 10
    with open('results/ticks.csv', 'w', newline='', encoding='utf-8') as f:
        wcsv = csv.writer(f)
        wcsv.writerow(['arm', 'tick', 'phase', 'err_ema', 'C', 'energy', 'approach', 'syncopes'])
        for arm in ARMS:
            per_seed = [np.array(r['tick_rows'][::ds]) for r in runs[arm]]
            nmin = min(len(a) for a in per_seed)
            rows = np.stack([a[:nmin] for a in per_seed]).mean(0)  # (nmin, 6)
            for i in range(nmin):
                wcsv.writerow([arm, i * ds, int(rows[i, 0])] + [f"{v:.5f}" for v in rows[i, 1:]])

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        colors = {'full': 'tab:red', 'uniform': 'tab:orange', 'none': 'tab:gray'}
        bounds, names, cur = [], [], 0
        for (n, l, k, rr) in FULL:
            bounds.append(cur)
            names.append(n)
            cur += int(round(l * fac))
        bounds.append(cur)
        for arm in ARMS:
            data = np.genfromtxt('results/ticks.csv', delimiter=',', names=True,
                                 dtype=None, encoding='utf-8')
            sel = data[data['arm'] == arm]
            t = sel['tick']
            axs[0, 0].semilogy(t, np.maximum(sel['err_ema'], 1e-6), color=colors[arm], label=arm)
            axs[0, 1].plot(t, sel['C'], color=colors[arm], label=arm)
            axs[1, 0].plot(t, sel['energy'], color=colors[arm], label=arm)
        for ax in [axs[0, 0], axs[0, 1], axs[1, 0]]:
            for b, nm in zip(bounds, names):
                ax.axvline(b, color='k', lw=0.4, alpha=0.5)
            axs[0, 0].set_title('live prediction error (EMA)')
            axs[0, 1].set_title('cognitive capital C')
            axs[1, 0].set_title('energy')
            axs[1, 0].set_xlabel('tick')
        axs[0, 0].legend()
        for b, nm in zip(bounds, names):
            axs[0, 0].text(b, axs[0, 0].get_ylim()[1], nm, fontsize=7, va='top')
        width = 0.25
        xs = np.arange(2)
        for ai, arm in enumerate(ARMS):
            v1, s1 = m(arm, 'rel_gain1'); v2, s2 = m(arm, 'rel_gain2')
            axs[1, 1].bar(xs + (ai - 1) * width, [v1, v2], width,
                          yerr=[s1, s2], capsize=3, color=colors[arm], label=arm)
        axs[1, 1].set_xticks(xs, ['idle gain I1 (W1→W2)', 'idle gain I2 (W3→W4)'])
        axs[1, 1].axhline(0, color='k', lw=0.5)
        axs[1, 1].set_title('relative error reduction across idle phases')
        axs[1, 1].legend()
        fig.suptitle('E0: does internal work during no-input phases create new prediction?')
        fig.tight_layout()
        fig.savefig('results/figure_e0.png', dpi=130)
        say('figure -> results/figure_e0.png')
    except Exception as e:
        say(f'(figure skipped: {e})')


if __name__ == '__main__':
    main()
