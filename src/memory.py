"""Ring buffer of episodes. Key = kernel state at encoding time.
Layout per row: [h(32), a(2), body(2), next_ant(2), next_body(2)] -> 40 cols."""
import numpy as np


class EpisodicMemory:
    def __init__(self, capacity=3000, seed=0):
        self.cap = capacity
        self.rng = np.random.default_rng(seed)
        self.data = np.zeros((capacity, 40))
        self.cluster = np.full(capacity, -1, dtype=np.int64)
        self.t = np.zeros(capacity, dtype=np.int64)
        self.count = 0

    def add(self, h, a, body, next_ant, next_body, cluster, t):
        i = self.count % self.cap
        self.data[i] = np.concatenate([h, a, body, next_ant, next_body])
        self.cluster[i] = cluster
        self.t[i] = t
        self.count += 1

    def retrieve(self, h, thresh=0.62):
        if self.count == 0:
            return None
        keys = self.data[:self.count, 0:32]
        sims = (keys @ h) / (np.linalg.norm(keys, axis=1) * np.linalg.norm(h) + 1e-9)
        i = int(np.argmax(sims))
        if sims[i] <= thresh:
            return None
        return float(sims[i]), i

    def _idx_of(self, c):
        return np.flatnonzero(self.cluster[:self.count] == c)

    def exemplar(self, c):
        idx = self._idx_of(c)
        if len(idx) == 0:
            return None
        return int(self.rng.choice(idx))

    def sample_cluster(self, c, n):
        idx = self._idx_of(c)
        if len(idx) == 0:
            return None
        return self.rng.choice(idx, size=min(n, len(idx)), replace=len(idx) < n)

    def prune(self, frac):
        n = min(self.count, self.cap)
        k = int(n * frac)
        if k == 0:
            return
        kill = self.rng.choice(n, k, replace=False)
        keep = np.setdiff1d(np.arange(n), kill)
        self.data[:len(keep)] = self.data[keep]
        self.cluster[:len(keep)] = self.cluster[keep]
        self.t[:len(keep)] = self.t[keep]
        self.count = len(keep)
