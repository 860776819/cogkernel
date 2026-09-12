"""Wiring: world + kernel + memory + models + partition/bandit + policy.

Label audit (plan axiom 2): the only training signals entering the brain are
  (a) world-model self-supervised next-state prediction error,
  (b) dC predictor trained on measured changes of cognitive capital C,
  (c) replay bandit reinforced by internal error-drop measurements.
No reward, no 'food/danger' label, no body-urgency hardcode exists here.
"""
import zlib
from collections import deque

import numpy as np

from world import WormWorld
from kernel import Kernel
from memory import EpisodicMemory
from nets import MLP
from brain import ClusterTracker, ReplayBandit

IM_EVERY = 3
IM_H = 6
IM_GAMMA = 0.95
IDLE_BLOCK = 50
DC_DELAY = 150
C_EVERY = 40
N_CLUSTERS = 6


class Agent:
    def __init__(self, seed, arm, variant='baseline-a'):
        assert arm in ('full', 'uniform', 'none')
        assert variant in ('baseline-a', 'minus-recall', 'minus-value', 'minus-bridge')
        self.arm = arm
        self.variant = variant
        # M0-minus ablations (M0_MINUS_DESIGN.md); each removes exactly one
        # programmer-installed answer, verified against tag baseline-a:
        #   minus-recall : remove the %10 reminiscence cron + pulse injection (P3)
        #   minus-value  : remove imagined-dC objective, candidate set, eps (P2+P6)
        #   minus-bridge : remove syncope->prune/forget survival-knowledge bridge (P1)
        self.probe_hook = None  # experimenter-side observer; never touches the brain
        self.seed = seed
        self.rng = np.random.default_rng(zlib.crc32(f"{seed}|{arm}".encode()))
        self.world = WormWorld(seed)  # identical world across arms for a given seed
        self.kernel = Kernel(seed)
        self.memory = EpisodicMemory(3000, seed)
        self.wm = MLP(36, 24, 4, np.random.default_rng(seed), lr=0.03)
        self.dc = MLP(36, 16, 1, np.random.default_rng(seed + 1), lr=0.02)
        self.clusters = ClusterTracker(N_CLUSTERS, 4, np.random.default_rng(seed + 2))
        self.bandit = ReplayBandit(rng=np.random.default_rng(seed + 3))
        # unlearned fixed motor readout, used only by minus-value
        self.motor_R = np.random.default_rng(seed + 4).normal(0, 1.0 / np.sqrt(32), (2, 32))

        self.tick = 0
        self.pulse_vec = np.zeros(4)
        self.pulse_env = 0.0
        self.err_ema = 0.1
        self.C = -0.3
        self.approach_ema = 0.0
        self.syncopes = 0
        self.paralyses = 0
        self.pulses = 0

        self.recentX = np.zeros((128, 36))
        self.recentY = np.zeros((128, 4))
        self.recent_n = 0
        self.probeX = np.zeros((512, 36))
        self.probeY = np.zeros((512, 4))
        self.probe_n = 0
        self.body_hist = deque(maxlen=6)
        self.dc_queue = []
        self.cached_a = np.array([0.0, 0.5])
        self.next_decide = 0

        # idle bookkeeping
        self.idle_t = 0
        self.block_t = 0
        self.cur_c = None
        self.ex_i = None
        self.err_before = None
        self.cos_before = []
        self.cos_during = []
        # entropy audit: count matrices (4 dims x 8 bins) per phase type
        self.ent = {'wake': np.zeros((4, 8)), 'idle': np.zeros((4, 8))}

    # ---------------- helpers ----------------
    def _entropy_acc(self, x, phase_type):
        bins = np.clip((np.clip(x, 0, 1.19) / 0.15).astype(int), 0, 7)
        for d in range(4):
            self.ent[phase_type][d, bins[d]] += 1

    def _remember_sample(self, X, Y):
        i = self.recent_n % 128
        self.recentX[i], self.recentY[i] = X, Y
        self.recent_n += 1
        # reservoir for cognitive-capital probes (uniform over history)
        if self.probe_n < 512:
            self.probeX[self.probe_n], self.probeY[self.probe_n] = X, Y
        else:
            j = self.rng.integers(0, self.probe_n + 1)
            if j < 512:
                self.probeX[j], self.probeY[j] = X, Y
        self.probe_n += 1

    def probe_mse(self, c, n):
        idx = self.memory.sample_cluster(c, n)
        if idx is None:
            return None
        rows = self.memory.data[idx]
        X = np.concatenate([rows[:, 0:32], rows[:, 32:34], rows[:, 34:36]], 1).T
        Y = np.concatenate([rows[:, 36:38], rows[:, 38:40]], 1).T
        return self.wm.loss(X, Y)

    def _eval_C(self):
        pool = min(self.probe_n, 512)
        n = min(64, pool)
        if n < 32:
            return
        idx = self.rng.choice(pool, n, replace=False)
        mse = self.wm.loss(self.probeX[idx].T, self.probeY[idx].T)
        self.C += 0.3 * ((-mse) - self.C)

    def rollout_scores(self, h, x, A):
        """A:(J,2) candidate actions -> imagined dC sums (J,). Uses wm + kernel
        + dc forward only; imagined data never trains anything (plan axiom 5)."""
        J = A.shape[0]
        H = np.tile(h[:, None], (1, J))
        B = np.tile(x[2:4][:, None], (1, J))
        slope = np.zeros((2, J))
        s = np.zeros(J)
        zero_m = np.zeros((4, J))
        for k in range(IM_H):
            X = np.concatenate([H, A.T, B], 0)          # (36,J)
            P = self.wm.forward(X)                       # (4,J): next ant+body
            nb = P[2:4]
            slope = 0.5 * slope + 0.5 * (nb - B)
            F = np.concatenate([H, B, slope], 0)
            s += (IM_GAMMA ** k) * self.dc.forward(F)[0]
            H = self.kernel.step_batch(H, P, zero_m)
            B = nb
        return s

    def policy(self, h, x, eps):
        if self.tick < self.next_decide:
            return self.cached_a
        self.next_decide = self.tick + IM_EVERY
        base = self.cached_a
        J = 5
        A = np.clip(base + self.rng.normal(0, 0.5, (J, 2)), [-1, 0], [1, 1])
        A = np.concatenate([A, np.array([[0.0, 0.0], [0.0, 0.7], [0.6, 0.6], [-0.6, 0.6]])], 0)
        if self.rng.random() < eps or self.memory.count < 150:
            self.cached_a = A[self.rng.integers(0, len(A))]
            return self.cached_a
        s = self.rollout_scores(h, x, A)
        self.cached_a = A[int(np.argmax(s))]
        return self.cached_a

    # ---------------- wake ----------------
    def tick_wake(self, eps):
        w = self.world
        if w.frozen > 0:
            # syncope daze = unconscious: the kernel keeps running (axiom:
            # time always flows) but nothing is learned, encoded, or labeled
            x = np.array([0.0, 0.0, w.energy, w.integrity])
            self.kernel.step(x, self.pulse_vec * self.pulse_env)
            self.pulse_env *= 0.88
            w.step(np.array([0.0, 0.0]))
            self.tick += 1
            return self.err_ema
        x = w.sense()
        self._entropy_acc(x, 'wake')
        m = self.pulse_vec * self.pulse_env
        h = self.kernel.step(x, m)
        self.pulse_env *= 0.88
        self.last_x = x

        if self.variant == 'minus-value':
            # unlearned fixed readout of the continuous dynamics: no objective,
            # no candidate grammar, no exploration schedule, no schedules at all
            r = np.tanh(self.motor_R @ h)
            a = np.array([r[0], (r[1] + 1.0) / 2.0])
            self.cached_a = a
        else:
            a = self.policy(h, x, eps)
        self.approach_ema += 0.01 * (w.approach_index() - self.approach_ema)
        syncope, par = w.step(a)
        if syncope:
            self.syncopes += 1
            if self.variant != 'minus-bridge':
                # physics of starvation: it burns knowledge (plan section 4).
                # [minus-bridge] severs this hand-built survival/cognition bridge.
                self.memory.prune(0.10)
                self.wm.forget(0.08)
                self.dc.forget(0.08)
        if par:
            self.paralyses += 1

        x2 = w.sense()
        body, nb = x[2:4], x2[2:4]
        X = np.concatenate([h, a, body])
        Y = np.concatenate([x2[:2], x2[2:4]])
        pred = self.wm.forward(X[:, None])[:, 0]
        err = float(np.mean((pred - Y) ** 2))
        self.err_ema += 0.005 * (err - self.err_ema)
        self._remember_sample(X, Y)

        c = self.clusters.assign(np.concatenate([x2[:2], body]))
        self.cur_c = c
        self.clusters.observe_err(c, err)

        # wake = busy: sparse, distracted online updates. Idle (consolidation)
        # gets the concentrated budget -- that asymmetry is the hypothesis.
        if self.tick % 4 == 0 and self.recent_n >= 8:
            idx = self.rng.choice(min(self.recent_n, 128), 6, replace=False)
            self.wm.train_step(self.recentX[idx].T, self.recentY[idx].T)
        self.memory.add(h, a, body, x2[:2], x2[2:4], c, self.tick)

        if self.tick % C_EVERY == 0:
            self._eval_C()

        # dC predictor: delayed real labels only (wake phases)
        self.body_hist.append(body.copy())
        if self.tick % 5 == 0 and len(self.body_hist) >= 5:
            dbody = body - self.body_hist[0]
            self.dc_queue.append((self.tick + DC_DELAY,
                                  np.concatenate([h, body, dbody]), c, self.C))
        while self.dc_queue and self.dc_queue[0][0] <= self.tick:
            _, feats, cc, c_then = self.dc_queue.pop(0)
            label = self.C - c_then
            self.dc.train_step(feats[:, None], np.array([[label]]))
            self.clusters.dread[cc] += 0.05 * (label - self.clusters.dread[cc])

        # spontaneous reminiscence: a past state floats up unasked
        # [minus-recall] removes this cron entirely: no memory->h channel remains
        if (self.variant != 'minus-recall' and self.tick % 10 == 0
                and self.pulse_env < 0.3):
            r = self.memory.retrieve(h, thresh=0.90)
            if r is not None:
                _, i = r
                self.pulse_vec = self.memory.data[i, 36:40].copy()
                self.pulse_env = 1.0
                self.pulses += 1

        if self.probe_hook is not None:
            self.probe_hook(self)
        self.tick += 1
        return err

    # ---------------- idle ----------------
    def _start_block(self):
        active = self.clusters.active()
        if len(active) == 0 or self.arm == 'none':
            self.cur_c = None
            return
        if self.arm == 'full':
            phi = self.bandit.features(self.clusters, active)
            ci, _ = self.bandit.choose(phi)
            c = int(active[ci])
            self._phi, self._ci = phi, ci
        else:
            c = int(active[self.rng.integers(0, len(active))])
        self.cur_c = c
        ex = self.memory.exemplar(c)
        self.ex_i = ex
        if ex is not None:
            self.pulse_vec = self.memory.data[ex, 36:40].copy()
            self.pulse_env = 1.0
            self.cos_before.append(float(self.kernel.h[:, 0] @ self.memory.data[ex, 0:32] /
                                         (np.linalg.norm(self.kernel.h[:, 0]) *
                                          np.linalg.norm(self.memory.data[ex, 0:32]) + 1e-9)))
        self.err_before = self.probe_mse(c, 16)
        self.block_t = 0

    def _end_block(self):
        c = self.cur_c
        if c is None:
            return
        err_after = self.probe_mse(c, 16)
        if self.err_before is not None and err_after is not None:
            r = (self.err_before - err_after) / (self.err_before + 1e-6)
            if self.arm == 'full' and self.ex_i is not None:
                self.bandit.update(self._phi, self._ci, r)
                self.bandit.history.append((c, r))
        self.cur_c = None

    def tick_idle(self):
        w = self.world
        if self.idle_t % IDLE_BLOCK == 0:
            self._start_block()
        # sensory gating at rest: antennae contribute nothing (entropy ~ 0)
        x = np.array([0.0, 0.0, w.energy, w.integrity]) + self.rng.normal(0, 0.01, 4)
        self._entropy_acc(x, 'idle')
        m = self.pulse_vec * self.pulse_env
        h = self.kernel.step(x, m)
        self.pulse_env *= 0.88
        if self.ex_i is not None and 5 <= self.block_t <= 25:
            key = self.memory.data[self.ex_i, 0:32]
            self.cos_during.append(float(h @ key / (np.linalg.norm(h) * np.linalg.norm(key) + 1e-9)))
        # internally rehearsing real past episodes of the chosen cluster
        if self.cur_c is not None and self.block_t in (12, 24, 36, 48):
            idx = self.memory.sample_cluster(self.cur_c, 32)
            if idx is not None:
                rows = self.memory.data[idx]
                Xr = np.concatenate([rows[:, 0:32], rows[:, 32:34], rows[:, 34:36]], 1).T
                Yr = np.concatenate([rows[:, 36:38], rows[:, 38:40]], 1).T
                self.wm.train_step(Xr, Yr)
        if self.block_t % 20 == 0:
            self._eval_C()
        self.block_t += 1
        if self.block_t == IDLE_BLOCK:
            self._end_block()
        self.idle_t += 1
        self.tick += 1

    def cos_s4(self):
        if len(self.cos_during) == 0:
            return None, None, 0
        return (float(np.mean(self.cos_before)), float(np.mean(self.cos_during)),
                len(self.cos_during))
