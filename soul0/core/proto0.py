"""Proto-0: minimal existence organization (Soul-0 E0).

Three strictly separated layers. Nothing else exists in this file.

WORLD RULES (apply to any configuration; never repair the Proto):
  W1 inert conversion : each component decays into the inert pool m at rate lam
  W2 finite medium    : total mass a + b + m is conserved (N = 1)
  W3 erosion events   : uniform removal of a fraction rho of a component/both,
                        applied at experimenter-chosen times

PROTO INNATE STRUCTURE (the existence organization; carried at birth, never
learned, never optimized):
  P1 cross-catalyzed incorporation: each component is re-formed from the inert
     pool at rate k * a * b * m -- the process runs only while BOTH components
     are present. a's maintenance requires b; b's requires a.

OBSERVER INSTRUMENTS (measurement only; these numbers never enter any
derivative; there is no alive/dead flag anywhere):
  presence threshold theta, recovery / collapse classification, horizons.
"""
import numpy as np

# --- parameters: the numeric part of the innate structure (researcher-given) ---
LAM = 0.2   # W1 inert conversion rate
K = 3.0     # P1 incorporation rate
N = 1.0     # W2 total mass

# --- observer instruments ---
THETA = 0.05            # presence threshold (vs saddle a1 ~ 0.0792)
A_STABLE = 0.420782     # stable fixed point a2 (derived, pre-registered)
RECOVER_LEVEL = 0.5 * A_STABLE   # recovered = min(a,b) back above this


def deriv(state, lam=LAM, k=K, coupling='both'):
    """coupling: 'both' = innate structure intact (a needs b, b needs a);
    'b-only' = b's incorporation disabled (half-cut);
    None     = all incorporation disabled (control: relations cut)."""
    a, b, m = state
    if coupling == 'both':
        p = k * a * b * m
        da = -lam * a + p
        db = -lam * b + p
        dm = lam * (a + b) - 2 * p
    elif coupling == 'b-only':
        p = k * a * b * m
        da = -lam * a + p
        db = -lam * b
        dm = lam * (a + b) - p
    elif coupling is None:
        da = -lam * a
        db = -lam * b
        dm = lam * (a + b)
    else:
        raise ValueError(coupling)
    return np.array([da, db, dm])


def rk4_step(state, dt, **kw):
    k1 = deriv(state, **kw)
    k2 = deriv(state + 0.5 * dt * k1, **kw)
    k3 = deriv(state + 0.5 * dt * k2, **kw)
    k4 = deriv(state + dt * k3, **kw)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(state0, T, dt=0.01, events=(), coupling='both', lam=LAM, k=K):
    """Run from state0 for time T. events = [(t, rho, 'both'|'b'), ...]:
    at time t, remove fraction rho of the named component(s) (W3).
    Returns (ts, traj)."""
    n = int(round(T / dt))
    ts = np.linspace(0.0, T, n + 1)
    traj = np.empty((n + 1, 3))
    s = np.array(state0, dtype=float)
    traj[0] = s
    ev = sorted(events)
    ei = 0
    for i in range(1, n + 1):
        t = ts[i]
        while ei < len(ev) and ev[ei][0] <= t:
            _, rho, mode = ev[ei]
            idx = [0, 1] if mode == 'both' else [1]
            moved = 0.0
            for j in idx:
                moved += rho * s[j]
                s[j] *= (1.0 - rho)
            s[2] += moved          # W2: erosion is forced inert-conversion, mass conserved
            ei += 1
        s = rk4_step(s, dt, coupling=coupling, lam=lam, k=k)
        traj[i] = s
    return ts, traj
