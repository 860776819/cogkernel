"""Two-layer MLP, batch columns, manual backprop. Dependency-free on purpose
(small nets; math is the standard 2-layer MSE regression)."""
import numpy as np


class MLP:
    def __init__(self, din, dh, dout, rng, lr=0.02):
        self.W1 = rng.normal(0, 1.0 / np.sqrt(din), (dh, din))
        self.b1 = np.zeros((dh, 1))
        self.W2 = rng.normal(0, 1.0 / np.sqrt(dh), (dout, dh))
        self.b2 = np.zeros((dout, 1))
        self.W1i = self.W1.copy()
        self.W2i = self.W2.copy()
        self.lr = lr

    def forward(self, X):
        return self.W2 @ np.tanh(self.W1 @ X + self.b1) + self.b2

    def loss(self, X, Y):
        P = self.forward(X)
        return float(np.mean((P - Y) ** 2))

    def grads(self, X, Y):
        H = np.tanh(self.W1 @ X + self.b1)
        P = self.W2 @ H + self.b2
        D = 2.0 * (P - Y) / Y.size
        gW2 = D @ H.T
        gb2 = D.sum(1, keepdims=True)
        DH = self.W2.T @ D
        DZ = (1 - H * H) * DH
        gW1 = DZ @ X.T
        gb1 = DZ.sum(1, keepdims=True)
        return gW1, gb1, gW2, gb2, float(np.mean((P - Y) ** 2))

    def train_step(self, X, Y):
        gW1, gb1, gW2, gb2, l = self.grads(X, Y)
        self.W1 -= self.lr * gW1
        self.b1 -= self.lr * gb1
        self.W2 -= self.lr * gW2
        self.b2 -= self.lr * gb2
        return l

    def forget(self, lam):
        # physics of syncope: weights decay toward their natal values
        self.W1 = (1 - lam) * self.W1 + lam * self.W1i
        self.W2 = (1 - lam) * self.W2 + lam * self.W2i
