"""Finite-difference check of the manual backprop in nets.MLP."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np
from nets import MLP

rng = np.random.default_rng(0)
net = MLP(5, 7, 3, rng, lr=0.0)
X = rng.normal(0, 1, (5, 4))
Y = rng.normal(0, 1, (3, 4))
gW1, gb1, gW2, gb2, _ = net.grads(X, Y)

eps = 1e-6
worst = 0.0
for name, P, G in [('W1', net.W1, gW1), ('b1', net.b1, gb1),
                   ('W2', net.W2, gW2), ('b2', net.b2, gb2)]:
    it = np.nditer(P, flags=['multi_index'])
    while not it.finished:
        ix = it.multi_index
        old = P[ix]
        P[ix] = old + eps
        lp = net.loss(X, Y)
        P[ix] = old - eps
        lm = net.loss(X, Y)
        P[ix] = old
        num = (lp - lm) / (2 * eps)
        rel = abs(num - G[ix]) / (abs(num) + abs(G[ix]) + 1e-8)
        worst = max(worst, rel)
        it.iternext()

print(f"max relative gradient error: {worst:.2e}")
assert worst < 1e-4, "GRADCHECK FAILED"
print("GRADCHECK PASS")
