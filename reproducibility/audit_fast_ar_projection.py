"""Numerical geometry audit for the closed-form Collatz--Wielandt outer projection.

This does not use the empirical data. It draws feasible matrices from synthetic row ellipsoids
and loading boxes, then verifies that every sampled spectral radius lies inside the reported
closed-form outer bounds for a fixed finite trial-vector set.
"""
import numpy as np


def rho(B):
    return float(np.max(np.abs(np.linalg.eigvals(np.asarray(B, float)))))


def outer_bounds(Ahat, Sig, q, mhat, lo, hi, xs):
    K = Ahat.shape[0]
    L, U = 0.0, np.inf
    for x in xs:
        x = np.maximum(np.asarray(x, float), 1e-10)
        x /= x.sum()
        mt = float(mhat @ x)
        urs, lrs = [], []
        for z in range(K):
            center = float(Ahat[z] @ x)
            rad = float(np.sqrt(max(q * (x @ Sig[z] @ x), 0.0)))
            du = lo[z] if mt >= 0 else hi[z]
            dl = hi[z] if mt >= 0 else lo[z]
            ub = max(center + rad - du * mt, 0.0)
            lb = max(center - rad - dl * mt, 0.0)
            urs.append(ub / x[z])
            lrs.append(lb / x[z])
        U = min(U, max(urs))
        L = max(L, min(lrs))
    return L, U


def draw_ellipsoid(rng, S, q):
    K = S.shape[0]
    v = rng.standard_normal(K)
    v /= np.linalg.norm(v)
    r = rng.random() ** (1.0 / K)
    return np.linalg.cholesky(S) @ (v * np.sqrt(q) * r)


rng = np.random.default_rng(20260920)
K = 5
# Choose a comfortably positive center so many samples also satisfy B>=0.
Ahat = np.abs(rng.normal(0.22, 0.06, (K, K)))
mhat = rng.normal(0.04, 0.02, K)
lo = rng.uniform(0.05, 0.15, K)
hi = lo + rng.uniform(0.10, 0.25, K)
q = 3.0
Sig = []
for _ in range(K):
    M = rng.normal(size=(K, K))
    S = M @ M.T
    S *= 0.0008 / np.trace(S)
    Sig.append(S)

xs = [np.ones(K)]
for _ in range(63):
    xs.append(np.maximum(np.abs(rng.standard_normal(K)), 1e-8))
L, U = outer_bounds(Ahat, Sig, q, mhat, lo, hi, xs)

rhos = []
tries = 0
while len(rhos) < 10000 and tries < 500000:
    tries += 1
    d = rng.uniform(lo, hi)
    A = np.vstack([Ahat[z] + draw_ellipsoid(rng, Sig[z], q) for z in range(K)])
    B = A - np.outer(d, mhat)
    if np.min(B) < 0:
        continue
    rhos.append(rho(B))

rhos = np.asarray(rhos)
if len(rhos) < 1000:
    raise SystemExit(f"FAIL: too few feasible draws ({len(rhos)})")
viol = np.sum((rhos < L - 1e-10) | (rhos > U + 1e-10))
print(f"feasible draws: {len(rhos)}")
print(f"outer interval: [{L:.6f}, {U:.6f}]")
print(f"sampled rho range: [{rhos.min():.6f}, {rhos.max():.6f}]")
print(f"violations: {viol}")
if viol:
    raise SystemExit("FAIL: sampled feasible spectral radius outside outer interval")
print("FAST AR PROJECTION GEOMETRY AUDIT PASS")
