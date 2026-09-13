"""Physics only. Substance types ('nutrient'/'hazard') are simulator
bookkeeping; the brain never sees type identity -- only chemical intensity
at two antennae plus its own body readings, and the physical consequences
of contact (energy/integrity changes, synapse decay on syncope)."""
import numpy as np


class WormWorld:
    def __init__(self, seed, n=16):
        self.rng = np.random.default_rng(seed)
        self.n = n
        self.reset_fields()
        self.pos = self.rng.uniform(0, n, 2)
        self.theta = self.rng.uniform(0, 2 * np.pi)
        self.energy = 0.8
        self.integrity = 1.0
        self.paralyzed = 0
        self.frozen = 0
        self.t = 0
        # edible substrate: contact consumes it locally; it regrows slowly.
        # No labels -- just an economy that makes "park on food" impossible.
        self.depl = np.zeros((n, n))

    def reset_fields(self):
        r = self.rng
        self.nutrient = [(r.uniform(0, self.n, 2), 1.8, r.uniform(0.5, 0.8)) for _ in range(2)]
        self.hazard = [(r.uniform(0, self.n, 2), 1.2, r.uniform(0.6, 0.9)) for _ in range(3)]

    def _d_torus(self, a, b):
        d = np.abs(a - b)
        d = np.minimum(d, self.n - d)
        return float(np.sqrt((d ** 2).sum()))

    def field(self, p, blobs):
        v = 0.0
        for c, s, amp in blobs:
            v += amp * np.exp(-(self._d_torus(p, c) ** 2) / (2 * s * s))
        return min(v, 1.2)

    def capability(self):
        e = float(np.clip(self.energy * 2, 0, 1))
        return (0.3 + 0.7 * e) * (0.3 + 0.7 * self.integrity)

    def _noise_scale(self):
        # poor body state degrades sensing -- physical, observable, unlabeled
        return 0.04 + 0.18 * (1 - self.integrity) + 0.08 * (1 - self.energy)

    def _cell(self, p):
        return int(p[0]) % self.n, int(p[1]) % self.n

    def nutrient_at(self, p):
        return self.field(p, self.nutrient) * (1.0 - self.depl[self._cell(p)])

    def antennae(self):
        perp = np.array([-np.sin(self.theta), np.cos(self.theta)])
        l = self.pos + 0.6 * perp
        rr = self.pos - 0.6 * perp
        return (self.nutrient_at(l) + self.field(l, self.hazard),
                self.nutrient_at(rr) + self.field(rr, self.hazard))

    def sense(self):
        nl, nr = self.antennae()
        s = self._noise_scale()
        ant = np.clip(np.array([nl, nr]) + self.rng.normal(0, s, 2), 0, 1.2)
        body = np.clip(np.array([self.energy, self.integrity]) + self.rng.normal(0, s * 0.3, 2), 0, 1)
        return np.concatenate([ant, body])

    def approach_index(self):
        # ground-truth measurement for the EXPERIMENTER only; never enters the brain
        return self.nutrient_at(self.pos) - self.field(self.pos, self.hazard)

    def step(self, action):
        """action=(turn,thrust). Returns (syncope_happened, new_paralysis)."""
        turn = float(np.clip(action[0], -1, 1))
        thrust = float(np.clip(action[1], 0, 1))
        self.t += 1
        if self.frozen > 0:  # syncope: time passes, body shut down, no contact
            self.frozen -= 1
            return False, False
        if self.paralyzed > 0:
            self.paralyzed -= 1
            turn, thrust = 0.0, 0.0
        cap = self.capability()
        self.theta += turn * 0.5 * cap
        speed = 0.14 * thrust * cap
        self.pos = (self.pos + speed * np.array([np.cos(self.theta), np.sin(self.theta)])) % self.n
        active = self.paralyzed == 0
        if active:
            ch = self.field(self.pos, self.hazard)
            cn = self.nutrient_at(self.pos)
            self.energy = min(1.0, self.energy + 0.25 * cn - (0.0045 + 0.008 * thrust * cap))
            self.integrity = min(1.0, self.integrity + 0.0015 - 0.22 * ch)
            if cn > 0.01:
                c = self._cell(self.pos)
                self.depl[c] = min(1.0, self.depl[c] + 0.02)  # grazing depletes
        self.depl = np.maximum(0.0, self.depl - 0.0003)       # slow regrowth
        syncope = False
        if self.energy <= 0.0:
            syncope = True
            self.energy = 0.45
            self.frozen = 100
        par = self.integrity <= 0.0 and self.paralyzed == 0
        if par:
            self.paralyzed = 200
        return syncope, par
