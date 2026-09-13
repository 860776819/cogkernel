"""Fixed random recurrent core (reservoir). v0 plasticity lives in readouts;
the axiom under test is that this state runs every tick, input or not."""
import numpy as np


class Kernel:
    def __init__(self, seed, hdim=32, xdim=4):
        r = np.random.default_rng(seed)
        self.hdim = hdim
        self.Wh = r.normal(0, 1.05 / np.sqrt(hdim), (hdim, hdim))
        self.Wx = r.normal(0, 0.6 / np.sqrt(xdim), (hdim, xdim))
        self.Wm = r.normal(0, 0.8 / np.sqrt(xdim), (hdim, xdim))
        self.b = r.normal(0, 0.1, (hdim, 1))
        self.h = np.zeros((hdim, 1))

    def step_batch(self, H, X, M, g=1.0):
        """H:(hdim,N) X:(xdim,N) M:(xdim,N) -> next states (hdim,N)."""
        return np.tanh(self.Wh @ H + g * (self.Wx @ X) + self.Wm @ M + self.b)

    def step(self, x, m, g=1.0):
        """x:(4,) m:(4,) -> h:(32,)."""
        self.h = self.step_batch(self.h, x[:, None], m[:, None], g)
        return self.h[:, 0]
