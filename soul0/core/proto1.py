"""Soul-0 E1 / R1: activity-integral regulatory freedom (minimal implementation).

R1 dynamics (pre-registered constants, SOUL0_E1_REGULATION_DESIGN.md):
    p      = G(r)*k*a*b*m,   G(r) = 1 + eps*tanh(r/R0)   in [1-|eps|, 1+|eps|]
    da/dt  = -lam*a + p
    db/dt  = -lam*b + p
    dm/dt  = lam*(a+b) - 2*p            (a+b+m conserved for ANY r)
    dr/dt  = p - mu*r                   (source reads ONLY the actual flux)

Layer separation:
    WORLD          : lam (inert conversion), finite medium
    PROTO INNATE   : cross-catalysis k; the existence of r and its coupling
                     structure (mu, eps, r0) -- declared before experiments
    PROTO RUNTIME  : the actual value and history of r
    OBSERVER       : theta, classification, horizons (never touch dynamics)

Pre-registered constants (no post-hoc tuning):
    mu = 0.02; eps in {+0.25, 0, -0.25}; r0 = lam*A2/mu.
"""
import numpy as np

LAM = 0.2
K = 3.0
N = 1.0
A2 = 0.420782                    # E0 stable fixed point
MU = 0.02                        # pre-registered
R0 = LAM * A2 / MU               # pre-registered: E0 equilibrium flux scale (~4.2078)
EPSES = (0.25, 0.0, -0.25)       # pre-registered sign trio


def G(r, eps):
    return 1.0 + eps * np.tanh(r / R0)


def deriv(S, eps, lam=LAM, k=K, mu=MU):
    """S: (4,) or (4, N); rows a,b,m,r. Vectorized over columns."""
    a, b, m, r = S[0], S[1], S[2], S[3]
    p = G(r, eps) * k * a * b * m
    return np.vstack([-lam * a + p,
                      -lam * b + p,
                      lam * (a + b) - 2 * p,
                      p - mu * r])


def rk4(S, dt, eps):
    k1 = deriv(S, eps)
    k2 = deriv(S + 0.5 * dt * k1, eps)
    k3 = deriv(S + 0.5 * dt * k2, eps)
    k4 = deriv(S + dt * k3, eps)
    return S + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_batch(states0, T, dt, eps):
    """states0: (N,4) rows a,b,m,r -> ts (M+1,), traj (M+1, N, 4)."""
    states0 = np.atleast_2d(np.asarray(states0, dtype=float))
    S = states0.T.copy()
    n = int(round(T / dt))
    ts = np.linspace(0.0, T, n + 1)
    out = np.empty((n + 1, S.shape[1], 4))
    out[0] = S.T
    for i in range(1, n + 1):
        S = rk4(S, dt, eps)
        out[i] = S.T
    return ts, out


def simulate1(state0, T, dt, eps):
    ts, traj = simulate_batch(np.atleast_2d(state0), T, dt, eps)
    return ts, traj[:, 0, :]


def deriv_clamped(S, eps, lam=LAM, k=K, mu=MU):
    """Frozen-r diagnostic variant: r held constant (row 4 zeroed).
    DIAGNOSTIC ONLY -- never used as the E1 dynamics."""
    d = deriv(S, eps, lam, k, mu)
    d[3] = 0.0
    return d


def simulate_clamped(state0, T, dt, eps, r_const):
    """Frozen-r diagnostic run: r pinned at r_const."""
    st = np.atleast_2d(np.asarray(state0, dtype=float)).copy()
    st[0, 3] = r_const
    S = st.T.copy()
    n = int(round(T / dt))
    out = np.empty((n + 1, S.shape[1], 4))
    out[0] = S.T
    for i in range(1, n + 1):
        S = rk4(S, dt, eps)
        S[3, :] = r_const
        out[i] = S.T
    return out


def frozen_r_boundary(eps, r_const):
    """Clamped-r frozen section of the existence boundary (diagnostic).
    a1/a2 solve lam = G(r)*k*a*(1-2a). Returns (a1, a2, rho*) or None."""
    Gv = G(r_const, eps)
    disc = 1.0 - 8.0 * LAM / (Gv * K)
    if disc < 0:
        return None
    s = np.sqrt(disc)
    a1 = (1.0 - s) / 4.0
    a2 = (1.0 + s) / 4.0
    return a1, a2, 1.0 - a1 / a2
