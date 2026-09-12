"""Experience partitioning (online k-means) + the replay-allocation bandit.

Inquiry drive ('探究欲') is operationalized here as per-cluster LEARNING
PROGRESS (slow-minus-fast error EMA): attention goes to places where error is
actively dropping -- learnable structure -- not to raw novelty."""
import numpy as np


class ClusterTracker:
    def __init__(self, k=6, dim=4, rng=None):
        self.k = k
        self.dim = dim
        self.rng = rng
        self.cent = None
        self.err_fast = np.zeros(k)
        self.err_slow = np.zeros(k)
        self.dread = np.zeros(k)      # EMA of realized dC labels (negative = bad)
        self.recency = np.zeros(k)
        self.count = np.zeros(k, dtype=np.int64)

    def assign(self, x):
        if self.cent is None:
            self.cent = np.tile(x, (self.k, 1)) + self.rng.normal(0, 0.05, (self.k, self.dim))
        d = ((self.cent - x) ** 2).sum(1)
        c = int(np.argmin(d))
        self.cent[c] += 0.05 * (x - self.cent[c])
        self.count[c] += 1
        self.recency *= 0.995
        self.recency[c] = min(1.0, self.recency[c] + 0.05)
        return c

    def observe_err(self, c, err):
        self.err_fast[c] += 0.15 * (err - self.err_fast[c])
        self.err_slow[c] += 0.008 * (err - self.err_slow[c])

    def lp(self, c):
        return self.err_slow[c] - self.err_fast[c]

    def active(self, min_n=8):
        return np.flatnonzero(self.count >= min_n)


class ReplayBandit:
    """Learns WHICH experience cluster to rehearse internally. Reward = that
    cluster's prediction-error drop on its own past real episodes -- an
    internal signal; no external teacher."""

    def __init__(self, nfeat=5, tau=0.5, lr=0.15, rng=None):
        self.theta = np.zeros(nfeat)
        self.tau = tau
        self.lr = lr
        self.baseline = 0.0
        self.rng = rng or np.random.default_rng(0)
        self.history = []  # (chosen cluster, reward, pi of chosen)

    @staticmethod
    def features(tracker, active):
        """(n_active, 5): [1, z(err), z(lp), z(dread), z(recency)]"""
        err = tracker.err_fast[active]
        lp = np.array([tracker.lp(a) for a in active])
        dr = tracker.dread[active]
        rec = tracker.recency[active]

        def z(v):
            return (v - v.mean()) / (v.std() + 1e-6)
        return np.stack([np.ones(len(active)), z(err), z(lp), z(dr), z(rec)], 1)

    def probs(self, phi):
        logits = phi @ self.theta / self.tau
        logits -= logits.max()
        p = np.exp(logits)
        return p / p.sum()

    def choose(self, phi):
        p = self.probs(phi)
        return int(self.rng.choice(len(p), p=p)), p

    def update(self, phi, ci, r):
        r = float(np.clip(r, -1.0, 1.0))
        self.baseline += 0.1 * (r - self.baseline)
        p = self.probs(phi)
        grad = phi[ci] - (p[:, None] * phi).sum(0)
        self.theta += self.lr * (r - self.baseline) * grad
