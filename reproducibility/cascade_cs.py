"""
cascade_cs.py — Inference on the cascade-risk radius rho(B0) of a nonnegative reproduction matrix
under (possibly weak) separation of a common drive.  Implements the theory note:
  * Lemma 1 / Theorem 1 : exact Gaussian confidence ellipsoid for B, valid for every kappa>0
  * Theorem 2           : projection onto rho via Collatz-Wielandt duality (inner SOCP, outer simplex search)
  * Theorem 5           : two-proxy 2SLS and Anderson-Rubin inference when the drive is unobserved
Dependencies: numpy, scipy, cvxpy (CLARABEL).
"""
import numpy as np, cvxpy as cp
from scipy.stats import chi2, f as fdist, t as tdist
from scipy.linalg import cholesky
from scipy.optimize import minimize

# ----------------------------------------------------------------------------- basic linear algebra
def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None); return beta

def perron(B):
    """Perron root and left/right Perron vectors of a nonnegative matrix (sign-normalized)."""
    w, V = np.linalg.eig(B); i = int(np.argmax(w.real)); v = np.real(V[:, i]); v *= np.sign(v.sum() or 1)
    wl, Vl = np.linalg.eig(B.T); j = int(np.argmax(wl.real)); u = np.real(Vl[:, j]); u *= np.sign(u.sum() or 1)
    return float(w[i].real), np.abs(u), np.abs(v)

def spectral_radius(B):
    return float(np.max(np.abs(np.linalg.eigvals(np.asarray(B, float)))))

# ----------------------------------------------------------------------------- Lemma 1 / Theorem 1
def fit_rows(Y, H, f=None, add_const=True):
    """Row-wise OLS of Y (T x K) on [1, H, f]. Returns B_hat (K x K), c_hat (K,), residual variances,
    and per-row covariance matrices Sigma_z of the H-coefficients (eq. (10) form, with sigma_z estimated)."""
    T, K = Y.shape
    cols = [np.ones(T)] if add_const else []
    cols.append(H)
    if f is not None: cols.append(np.asarray(f).reshape(T, -1))
    X = np.column_stack(cols); p = X.shape[1]
    XtXi = np.linalg.inv(X.T @ X)
    off = 1 if add_const else 0
    Bh = np.zeros((K, K)); ch = np.zeros(K); s2 = np.zeros(K); Sig = []
    for z in range(K):
        th = ols(X, Y[:, z]); r = Y[:, z] - X @ th
        s2[z] = r @ r / (T - p); Bh[z] = th[off:off + K]
        ch[z] = th[-1] if f is not None else np.nan
        Sig.append(s2[z] * XtXi[off:off + K, off:off + K])
    return Bh, ch, s2, Sig

def separation_share(H, f):
    """kappa = ||f_perp||^2 / ||f_c||^2 and the aliasing direction gamma (Definition 1 of manuscript B)."""
    Hc = H - H.mean(0); fc = np.asarray(f, float) - np.mean(f)
    gamma = np.linalg.solve(Hc.T @ Hc, Hc.T @ fc); fperp = fc - Hc @ gamma
    return float(fperp @ fperp / (fc @ fc)), gamma, fperp

def ellipsoid_radius(K, T, p, alpha, mode="chi2"):
    """Joint radius q. 'chi2': known-sigma exact chi2_{K^2}; 'bonf': per-row K*F quantile at alpha/K
    (exact with estimated sigma under Gaussianity; use with per-row constraints)."""
    if mode == "chi2": return chi2.ppf(1 - alpha, K * K)
    return K * fdist.ppf(1 - alpha / K, K, T - p)

# ----------------------------------------------------------------------------- Theorem 2: CW projection
class ConeProjectedCS:
    """Confidence interval for rho(B0): image of S = {B>=0 : quadratic constraints} under rho,
    computed by Collatz-Wielandt duality.  mode='joint' uses one joint ellipsoid of radius q;
    mode='rows' uses per-row ellipsoids (Bonferroni), each of radius q."""
    def __init__(self, Bh, Sig_list, q, mode="joint", extra_cone=True):
        K = Bh.shape[0]; self.K = K; self.Bh = np.asarray(Bh, float)
        Lw = [np.linalg.inv(cholesky(S, lower=True)) for S in Sig_list]
        self.B = cp.Variable((K, K), nonneg=extra_cone); self.lam = cp.Variable(); self.x = cp.Parameter(K, nonneg=True)
        dev = [Lw[z] @ (self.B[z] - self.Bh[z]) for z in range(K)]
        cons = [cp.sum_squares(cp.hstack(dev)) <= q] if mode == "joint" else [cp.sum_squares(d) <= q for d in dev]
        self.plo = cp.Problem(cp.Minimize(self.lam), cons + [self.B @ self.x <= self.lam * self.x])
        self.phi = cp.Problem(cp.Maximize(self.lam), cons + [self.B @ self.x >= self.lam * self.x])
    def _inner(self, x, which):
        self.x.value = np.maximum(x, 1e-9) / np.sum(np.maximum(x, 1e-9))
        p = self.plo if which == "lo" else self.phi
        try:
            p.solve(solver="CLARABEL", warm_start=True)
        except Exception:
            return np.inf if which == "lo" else -np.inf
        if p.status == "unbounded": return -np.inf if which == "hi" else np.inf
        if p.status not in ("optimal", "optimal_inaccurate"): return np.inf if which == "lo" else -np.inf
        return float(self.lam.value)
    def _outer(self, which, restarts, iters, rng):
        K = self.K
        def obj(t):
            x = np.exp(t - t.max()); x /= x.sum(); v = self._inner(x, which); return v if which == "lo" else -v
        _, u, v = perron(np.maximum(self.Bh, 0))
        seeds = [np.log(np.maximum(v, 1e-6)), np.zeros(K)] + [rng.standard_normal(K) for _ in range(max(0, restarts - 2))]
        best = np.inf
        for t0 in seeds:
            r = minimize(obj, t0, method="Nelder-Mead", options={"maxiter": iters, "xatol": 1e-3, "fatol": 1e-6})
            best = min(best, r.fun)
        return best if which == "lo" else -best
    def interval(self, restarts=4, iters=80, seed=0):
        rng = np.random.default_rng(seed)
        lo = self._outer("lo", restarts, iters, rng); hi = self._outer("hi", restarts, iters, rng)
        return float(lo), (float(hi) if np.isfinite(hi) else np.inf)

def cascade_cs(Y, H, f, alpha=0.05, mode="rows", **kw):
    """One-call interface: drive-corrected fit + cone-projected confidence interval for rho(B0).
    Returns dict with rho_hat (at the cone-projected estimate), the interval, kappa, gamma."""
    T, K = Y.shape
    Bh, ch, s2, Sig = fit_rows(Y, H, f)
    kappa, gamma, _ = separation_share(H, f)
    p = 2 + K
    q = ellipsoid_radius(K, T, p, alpha, "bonf" if mode == "rows" else "chi2")
    lo, hi = ConeProjectedCS(Bh, Sig, q, mode=mode).interval(**kw)
    return dict(B_hat=Bh, c_hat=ch, rho_hat=spectral_radius(np.maximum(Bh, 0)), ci=(lo, hi), kappa=kappa, gamma=gamma, q=q)

# ----------------------------------------------------------------------------- Theorem 5: two proxies
def two_stage_ls(Y, H, m1, m2):
    """Row-wise 2SLS: regressors [1,H,m1], instruments [1,H,m2]. Returns B_2sls, c_2sls, first-stage F."""
    T, K = Y.shape
    X = np.column_stack([np.ones(T), H, m1]); Z = np.column_stack([np.ones(T), H, m2])
    Pi = ols(Z, X); Xhat = Z @ Pi
    fs = ols(Z, m1); res = m1 - Z @ fs; s2 = res @ res / (T - Z.shape[1])
    F = fs[-1] ** 2 / (s2 * np.linalg.inv(Z.T @ Z)[-1, -1])
    B2 = np.zeros((K, K)); c2 = np.zeros(K)
    for z in range(K):
        th = ols(Xhat, Y[:, z]); B2[z] = th[1:K + 1]; c2[z] = th[-1]
    return B2, c2, float(F)

def anderson_rubin_set(y, H, m1, m2, grid, alpha=0.05):
    """AR confidence set for the drive coefficient (scale of m1) of ONE row: {c0 : |t_{m2}| <= t_crit}
    in the OLS of (y - c0*m1) on [1,H,m2]. Exact t under Gaussianity for any first-stage strength."""
    T = len(y); Z = np.column_stack([np.ones(T), H, m2]); ZtZi = np.linalg.inv(Z.T @ Z); p = Z.shape[1]
    tc = tdist.ppf(1 - alpha / 2, T - p); acc = []
    for c0 in grid:
        yy = y - c0 * m1; b = ols(Z, yy); r = yy - Z @ b; s2 = r @ r / (T - p)
        acc.append(abs(b[-1] / np.sqrt(s2 * ZtZi[-1, -1])) <= tc)
    acc = np.array(acc); return grid[acc], acc

# ----------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    rng = np.random.default_rng(0); K, T = 4, 1500
    B0 = np.abs(rng.normal(0, .3, (K, K))) * (rng.random((K, K)) < .7); np.fill_diagonal(B0, .05); B0 *= .65 / spectral_radius(B0)
    H = rng.standard_normal((T, K)); Hc = H - H.mean(0); g0 = .25 * rng.standard_normal(K)
    v = rng.standard_normal(T); v -= v.mean(); v -= Hc @ np.linalg.solve(Hc.T @ Hc, Hc.T @ v); v *= 2.0 / np.linalg.norm(v)   # weak: ||f_perp||^2=4
    f = Hc @ g0 + v; c = rng.uniform(.6, 1.4, K)
    Y = Hc @ B0.T + np.outer(f, c) + rng.standard_normal((T, K))
    out = cascade_cs(Y, H, f, alpha=0.05, mode="rows")
    print(f"self-test: kappa={out['kappa']:.4f} rho0={spectral_radius(B0):.3f} rho_hat={out['rho_hat']:.3f} CI={tuple(round(x,3) for x in out['ci'])}  covers={out['ci'][0]<=spectral_radius(B0)<=out['ci'][1]}")
    m1 = f + .7 * rng.standard_normal(T); m2 = f + .7 * rng.standard_normal(T)
    B2, c2, F = two_stage_ls(Y, H, m1, m2); print(f"2SLS first-stage F={F:.1f}; |bias B|={np.linalg.norm(B2-B0):.3f}")
    s, _ = anderson_rubin_set(Y[:, 0], H, m1, m2, np.linspace(-4, 6, 501)); print(f"AR set row0: [{s.min():.2f},{s.max():.2f}] contains c0={c[0]:.2f}: {s.min()<=c[0]<=s.max()}")

# ----------------------------------------------------------------------------- certified OUTER bounds
def certified_interval(Bh, Sig_list, q, mode="rows", grid=None, n_grid=300, seed=0):
    """Certified OUTER bounds [L,U] ⊇ rho(S) via Collatz–Wielandt: for ANY x>0,
       rho(B) <= max_i (Bx)_i/x_i  and  rho(B) >= min_i (Bx)_i/x_i  (B>=0),
       so U(x)=max_i max_{B in S}(Bx)_i/x_i and L(x)=min_i min_{B in S}(Bx)_i/x_i bound rho+(S), rho-(S) for every x;
       minimizing U and maximizing L over a simplex grid only tightens. Each U(x), L(x) is K linear programs.
       Returns (L, U) or None if S is empty (LP infeasible)."""
    K = Bh.shape[0]; rng = np.random.default_rng(seed)
    Lw = [np.linalg.inv(cholesky(S, lower=True)) for S in Sig_list]
    B = cp.Variable((K, K), nonneg=True); x = cp.Parameter(K, nonneg=True)
    dev = [Lw[z] @ (B[z] - Bh[z]) for z in range(K)]
    cons = [cp.sum_squares(cp.hstack(dev)) <= q] if mode == "joint" else [cp.sum_squares(d) <= q for d in dev]
    pU = [cp.Problem(cp.Maximize((B @ x)[i]), cons) for i in range(K)]
    pL = [cp.Problem(cp.Minimize((B @ x)[i]), cons) for i in range(K)]
    if grid is None:
        grid = rng.dirichlet(np.ones(K), n_grid)
        _, u, v = perron(np.maximum(Bh, 0)); grid = np.vstack([grid, v / v.sum(), np.ones(K) / K])
    Ubest, Lbest = np.inf, -np.inf
    for xv in grid:
        xv = np.maximum(xv, 1e-6); xv /= xv.sum(); x.value = xv
        vals = []
        for i in range(K):
            pU[i].solve(solver="CLARABEL", warm_start=True)
            if pU[i].status == "infeasible": return None
            vals.append(pU[i].value / xv[i])
        Ubest = min(Ubest, max(vals))
        vals = []
        for i in range(K):
            pL[i].solve(solver="CLARABEL", warm_start=True); vals.append(pL[i].value / xv[i])
        Lbest = max(Lbest, min(vals))
    return float(Lbest), float(Ubest)

# ----------------------------------------------------------------------------- certified OUTER interval
def certified_interval(Bh, Sig_list, q, mode="rows", n_restarts=6, iters=120, grid=None, seed=0):
    """OUTER (conservative) bounds on rho over S = {B>=0: quadratic constraints}, hence a reported interval whose
    coverage is inherited from the coverage of S.  Collatz-Wielandt: for any x>0,
        rho^+(S) <= U(x) := max_{B in S} max_i (Bx)_i/x_i,   rho^-(S) >= L(x) := min_{B in S} min_i (Bx)_i/x_i,
    each U(x), L(x) a set of K convex programs. x is optimized on the simplex (Nelder-Mead from several starts; for
    K<=3 an optional grid certifies the optimum). Returns (L*, U*), or None if S is empty."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed); K = Bh.shape[0]
    Lw = [np.linalg.inv(cholesky(S, lower=True)) for S in Sig_list]
    B = cp.Variable((K, K), nonneg=True); dev = [Lw[z] @ (B[z] - Bh[z]) for z in range(K)]
    cons = [cp.sum_squares(cp.hstack(dev)) <= q] if mode == "joint" else [cp.sum_squares(d) <= q for d in dev]
    fe = cp.Problem(cp.Minimize(0), cons); fe.solve(solver="CLARABEL")
    if fe.status in ("infeasible", "infeasible_inaccurate"): return None
    x = cp.Parameter(K, nonneg=True)
    probU = [cp.Problem(cp.Maximize((B @ x)[i]), cons) for i in range(K)]
    probL = [cp.Problem(cp.Minimize((B @ x)[i]), cons) for i in range(K)]
    def U(xv):
        xv = np.maximum(xv, 1e-9); xv = xv / xv.sum(); x.value = xv
        return max(p.solve(solver="CLARABEL", warm_start=True) / xv[i] for i, p in enumerate(probU))
    def L(xv):
        xv = np.maximum(xv, 1e-9); xv = xv / xv.sum(); x.value = xv
        return min(p.solve(solver="CLARABEL", warm_start=True) / xv[i] for i, p in enumerate(probL))
    _, u, v = perron(np.maximum(Bh, 0)); seeds = [np.log(np.maximum(v, 1e-6)), np.zeros(K)] + [rng.standard_normal(K) for _ in range(n_restarts - 2)]
    soft = lambda t: np.exp(t - t.max()) / np.exp(t - t.max()).sum()
    Ub = min(minimize(lambda t: U(soft(t)), t0, method="Nelder-Mead", options={"maxiter": iters, "xatol": 1e-3, "fatol": 1e-6}).fun for t0 in seeds)
    Lb = max(-minimize(lambda t: -L(soft(t)), t0, method="Nelder-Mead", options={"maxiter": iters, "xatol": 1e-3, "fatol": 1e-6}).fun for t0 in seeds)
    if grid is not None:   # simplex grid certification (small K)
        for xv in grid: Ub = min(Ub, U(xv)); Lb = max(Lb, L(xv))
    return max(Lb, 0.0), Ub

def simplex_grid(K, n):
    """all compositions of n into K parts, normalized (K<=4 recommended)"""
    import itertools
    pts = []
    for comb in itertools.combinations(range(n + K - 1), K - 1):
        parts = np.diff([-1] + list(comb) + [n + K - 1]) - 1
        if (parts > 0).all(): pts.append(parts / n)
    return pts

# ----------------------------------------------------------------------------- Paper 2 additions
def connectedness(B):
    """Total connectedness (Diebold-Yilmaz-type, %) and systemic contributions (column sums minus one) of the
    all-generation response matrix R=(I-B)^{-1} of a nonnegative reproduction matrix with rho(B)<1."""
    B = np.maximum(np.asarray(B, float), 0); K = B.shape[0]
    if spectral_radius(B) >= 1: return dict(total=np.nan, systemic=np.full(K, np.nan), R=None)
    R = np.linalg.inv(np.eye(K) - B); S = R / R.sum(1, keepdims=True)
    return dict(total=100 * (1 - np.trace(S) / K), systemic=R.sum(0) - 1, R=R)

def sargan_j(y, H, m1, instruments):
    """Sargan-Hansen overidentification statistic for one equation: regressors [1,H,m1], instruments [1,H,instruments].
    Returns (J, p-value, df). Unreliable under weak instruments (use Kleibergen 2005 variants)."""
    from scipy.stats import chi2 as _chi2
    T = len(y); X = np.column_stack([np.ones(T), H, m1]); Z = np.column_stack([np.ones(T), H] + [np.asarray(i).reshape(T) for i in instruments])
    Xh = Z @ ols(Z, X); th = ols(Xh, y); u = y - X @ th
    b = ols(Z, u); R2 = 1 - np.sum((u - Z @ b) ** 2) / np.sum((u - u.mean()) ** 2); df = len(instruments) - 1
    J = T * R2; return float(J), float(1 - _chi2.cdf(J, df)), df

def ar_projected_interval(Y, H, m1, m2, alpha=0.05, nx=32, seed=0, hac_lags=0):
    """Fast conservative certified outer interval for cascade risk using analytic Anderson--Rubin inversion.

    This routine uses exactly the same rowwise AR loading sets and reduced-form history-coefficient
    ellipsoids as the previous implementation, but replaces the repeated CVXPY Collatz--Wielandt
    support solves by closed-form ellipsoidal support functions.

    For every trial vector x>0 and every feasible row z,

        B_z x = A_z x - d_z (m^(1)' x).

    Over the row ellipsoid, max/min A_z x are available in closed form.  Dropping B_z>=0 only for
    these support evaluations enlarges the feasible set, so the resulting upper bound cannot be too
    small; the lower support is additionally truncated at zero, which cannot exceed the true support
    because every feasible B is nonnegative.  Hence [L,U] remains a certified OUTER interval.

    A single feasibility SOCP is retained to detect an empty confidence set.  Thus a bounded design
    requires one conic solve rather than O(nx*K) solves.  No grid truncation or arbitrary coefficient
    cap is used.  If any positive AR loading set is unbounded, the routine returns [0,+inf].

    H may contain the K network-history columns followed by extra conditioning columns.
    """
    from scipy.stats import f as _fdist

    Y = np.asarray(Y, float)
    H = np.asarray(H, float)
    m1 = np.asarray(m1, float).reshape(-1)
    m2 = np.asarray(m2, float).reshape(-1)
    T, K = Y.shape
    a1 = a2 = alpha / 2.0

    Hc = H - H.mean(0)
    Xn = np.column_stack([np.ones(T), Hc])
    df = T - Xn.shape[1]
    if df <= K:
        raise ValueError(
            f"Insufficient residual degrees of freedom: T={T}, H columns={H.shape[1]}, K={K}"
        )

    HtH = Hc.T @ Hc
    HtHi = np.linalg.pinv(HtH)
    Ahat = np.vstack([ols(Xn, Y[:, z])[1:K + 1] for z in range(K)])
    m1hat = (HtHi @ Hc.T @ (m1 - m1.mean()))[:K]

    lo = np.zeros(K)
    hi = np.zeros(K)
    bounded = np.zeros(K, dtype=bool)
    sbar2 = np.zeros(K)
    unbounded_seen = False

    for z in range(K):
        kind, pieces = anderson_rubin_analytic(
            Y[:, z], Hc, m1, m2,
            alpha=a2 / K,
            hac_lags=hac_lags,
        )
        boxes = ar_pieces_to_dboxes(kind, pieces)
        if not boxes:
            return dict(
                lo=np.nan,
                hi=np.nan,
                ar_bounded=bounded,
                empty=True,
                projection='fast_closed_form_outer',
            )

        lo[z] = min(b[0] for b in boxes)
        if any(b[1] is None for b in boxes):
            bounded[z] = False
            unbounded_seen = True
            continue

        hi[z] = max(float(b[1]) for b in boxes)
        bounded[z] = True

        vals = []
        for blo, bhi in boxes:
            for c0 in (float(blo), float(bhi)):
                yy = Y[:, z] - c0 * m1
                rr = yy - Xn @ ols(Xn, yy)
                vals.append(float(rr @ rr / df))
        sbar2[z] = max(vals)

    if unbounded_seen:
        return dict(
            lo=0.0,
            hi=np.inf,
            ar_bounded=bounded,
            empty=False,
            projection='fast_closed_form_outer',
        )

    Vhist = HtHi[:K, :K]
    Vhist = 0.5 * (Vhist + Vhist.T)
    qA = K * _fdist.ppf(1 - a1 / K, K, df)

    # Row covariance matrices for the reduced-form history coefficients.
    Sig = []
    for z in range(K):
        Sz = max(sbar2[z], 1e-14) * Vhist
        Sz = 0.5 * (Sz + Sz.T) + 1e-12 * np.eye(K)
        Sig.append(Sz)

    # One feasibility solve only.  This preserves the previous 'empty set' diagnostic.
    # The expensive repeated support solves are eliminated below.
    A = cp.Variable((K, K))
    d = cp.Variable(K)
    Bv = A - cp.reshape(d, (K, 1), order='C') @ np.reshape(m1hat, (1, K))
    cons = [Bv >= 0, d >= lo, d <= hi]
    for z in range(K):
        Lw = np.linalg.inv(cholesky(Sig[z], lower=True))
        cons.append(cp.sum_squares(Lw @ (A[z] - Ahat[z])) <= qA)

    fe = cp.Problem(cp.Minimize(0), cons)
    try:
        fe.solve(solver='CLARABEL', warm_start=True)
    except Exception:
        # Solver failure is not evidence that the statistical confidence set is empty.
        # Return the conservative uninformative outer interval rather than a false rejection.
        return dict(
            lo=0.0,
            hi=np.inf,
            ar_bounded=bounded,
            empty=False,
            projection='fast_closed_form_outer_solver_fallback',
        )

    if fe.status in ('infeasible', 'infeasible_inaccurate'):
        return dict(
            lo=np.nan,
            hi=np.nan,
            ar_bounded=bounded,
            empty=True,
            projection='fast_closed_form_outer',
        )
    if fe.status not in ('optimal', 'optimal_inaccurate'):
        return dict(
            lo=0.0,
            hi=np.inf,
            ar_bounded=bounded,
            empty=False,
            projection='fast_closed_form_outer_solver_fallback',
        )

    # Trial vectors.  Validity does not depend on how these are chosen; additional vectors only tighten.
    # Use both the reduced-form center and a midpoint loading-corrected center before deterministic random starts.
    xs = []
    try:
        _, _, vA = perron(np.maximum(Ahat, 0))
        xs.append(np.maximum(vA, 1e-8))
    except Exception:
        pass

    dmid = 0.5 * (lo + hi)
    Bmid = np.maximum(Ahat - np.outer(dmid, m1hat), 0)
    try:
        _, _, vB = perron(Bmid)
        xs.append(np.maximum(vB, 1e-8))
    except Exception:
        pass

    xs.append(np.ones(K))
    rng = np.random.default_rng(seed)
    while len(xs) < max(3, int(nx)):
        xs.append(np.maximum(np.abs(rng.standard_normal(K)), 1e-8))

    U = np.inf
    L = 0.0

    for x in xs:
        x = np.maximum(np.asarray(x, float), 1e-10)
        x = x / x.sum()
        mt = float(m1hat @ x)

        upper_ratio = np.empty(K)
        lower_ratio = np.empty(K)

        for z in range(K):
            center = float(Ahat[z] @ x)
            rad = float(np.sqrt(max(qA * (x @ Sig[z] @ x), 0.0)))

            # max_d[-d*mt] and min_d[-d*mt] on d in [lo_z, hi_z]
            d_for_upper = lo[z] if mt >= 0 else hi[z]
            d_for_lower = hi[z] if mt >= 0 else lo[z]

            ub_num = center + rad - d_for_upper * mt
            lb_num_relaxed = center - rad - d_for_lower * mt

            # Feasible B is nonnegative; relaxing the cone for support calculations is conservative.
            ub_num = max(ub_num, 0.0)
            lb_num = max(lb_num_relaxed, 0.0)

            upper_ratio[z] = ub_num / x[z]
            lower_ratio[z] = lb_num / x[z]

        U = min(U, float(np.max(upper_ratio)))
        L = max(L, float(np.min(lower_ratio)))

    return dict(
        lo=max(float(L), 0.0),
        hi=(float(U) if np.isfinite(U) else np.inf),
        ar_bounded=bounded,
        empty=False,
        projection='fast_closed_form_outer',
    )

# ----------------------------------------------------------------------------- Review-3 corrections: analytic AR, HAC, MOP F, Hansen J
def _partial(y, X):
    """residual of y on X (with intercept in X)"""
    return y - X @ ols(X, y)

def _hac_lrv(g, L):
    """Newey-West long-run variance of a scalar series g (Bartlett kernel, bandwidth L)"""
    g = g - g.mean(); s = g @ g
    for l in range(1, L + 1): s += 2 * (1 - l / (L + 1)) * (g[l:] @ g[:-l])
    return s

def anderson_rubin_analytic(y, H, m1, m2, alpha=0.05, hac_lags=0):
    """EXACT inversion of the Anderson-Rubin test for H0: c = c0 with one endogenous regressor (m1) and one
    instrument (m2), after partialling [1,H].  |t(c0)| <= t_crit  <=>  Q(c0) = a c0^2 + b c0 + k <= 0, a quadratic.
    hac_lags=0: homoskedastic t (exact t_{T-K-2} under Gaussianity); hac_lags>0: Newey-West variance (asymptotic),
    still quadratic in c0 because the residual is affine in c0.
    Returns (kind, pieces): kind in {'interval','all','complement','empty'}, pieces = list of (lo,hi) with +-inf."""
    from scipy.stats import t as _t, norm as _norm
    T = len(y); X = np.column_stack([np.ones(T), H]); p = X.shape[1] + 1
    yt = _partial(y, X); m1t = _partial(m1, X); m2t = _partial(m2, X); s22 = m2t @ m2t
    # b(c0) = (m2t'yt - c0 m2t'm1t)/s22  =: (b0 - c0 b1)/s22 ; u(c0) = yt - c0 m1t - b(c0) m2t (affine in c0)
    b0 = m2t @ yt; b1 = m2t @ m1t
    crit2 = (_t.ppf(1 - alpha / 2, T - p) if hac_lags == 0 else _norm.ppf(1 - alpha / 2)) ** 2
    def Q(c0):
        b = (b0 - c0 * b1) / s22; u = yt - c0 * m1t - b * m2t
        if hac_lags == 0: V = (u @ u) / (T - p) / s22
        else: V = _hac_lrv(m2t * u, hac_lags) / s22 ** 2
        return b * b - crit2 * V
    # Q is exactly quadratic: fit through three points
    c = np.array([-1.0, 0.0, 1.0]); q = np.array([Q(x) for x in c]); A = np.vstack([c ** 2, c, np.ones(3)]).T
    a2, a1, a0 = np.linalg.solve(A, q)
    if abs(a2) < 1e-14 * max(1.0, abs(a1), abs(a0)):
        if abs(a1) < 1e-14: return ('all', [(-np.inf, np.inf)]) if a0 <= 0 else ('empty', [])
        r = -a0 / a1; return ('interval', [(-np.inf, r)]) if a1 > 0 else ('interval', [(r, np.inf)])
    disc = a1 * a1 - 4 * a2 * a0
    if disc < 0: return ('all', [(-np.inf, np.inf)]) if a2 < 0 else ('empty', [])
    r1, r2 = sorted(((-a1 - np.sqrt(disc)) / (2 * a2), (-a1 + np.sqrt(disc)) / (2 * a2)))
    if a2 > 0: return ('interval', [(r1, r2)])          # Q<=0 between roots
    return ('complement', [(-np.inf, r1), (r2, np.inf)])   # Q<=0 outside roots

def effective_F(m1, m2, H, hac_lags=0):
    """First-stage F for one instrument after partialling [1,H]; hac_lags>0 gives the Newey-West-robust version,
    which for a single instrument coincides with the Montiel Olea-Pflueger (2013) effective F."""
    T = len(m1); X = np.column_stack([np.ones(T), H]); m1t = _partial(m1, X); m2t = _partial(m2, X)
    pi = (m2t @ m1t) / (m2t @ m2t); u = m1t - pi * m2t
    V = (u @ u) / (T - X.shape[1] - 1) / (m2t @ m2t) if hac_lags == 0 else _hac_lrv(m2t * u, hac_lags) / (m2t @ m2t) ** 2
    return float(pi * pi / V)

def hansen_j_hac(y, H, m1, instruments, hac_lags=12):
    """Hansen (1982) J with HAC weighting (two-step GMM), one endogenous regressor, instruments after partialling [1,H].
    Returns (J, p, df). Diagnostic only under weak instruments."""
    from scipy.stats import chi2 as _chi2
    T = len(y); X = np.column_stack([np.ones(T), H]); yt = _partial(y, X); m1t = _partial(m1, X)
    Z = np.column_stack([_partial(np.asarray(z).reshape(T), X) for z in instruments]); q = Z.shape[1]
    # step 1: 2SLS
    Pz = Z @ np.linalg.solve(Z.T @ Z, Z.T @ m1t); c1 = (Pz @ yt) / (Pz @ m1t); u = yt - c1 * m1t
    def S_hat(u):
        G = Z * u[:, None]; S = G.T @ G
        for l in range(1, hac_lags + 1): w = 1 - l / (hac_lags + 1); C = G[l:].T @ G[:-l]; S += w * (C + C.T)
        return S / T
    # step 2: efficient GMM
    W = np.linalg.inv(S_hat(u)); Zm = Z.T @ m1t / T; Zy = Z.T @ yt / T
    c2 = (Zm @ W @ Zy) / (Zm @ W @ Zm); u2 = yt - c2 * m1t; g = Z.T @ u2 / T
    J = T * g @ np.linalg.inv(S_hat(u2)) @ g; df = q - 1
    return float(J), float(1 - _chi2.cdf(J, df)), df

def ar_pieces_to_dboxes(kind, pieces):
    """AR set ∩ R+  ->  list of (lo, hi) boxes for the loading d (hi=None means unbounded above)."""
    boxes = []
    for lo, hi in pieces:
        lo2 = max(0.0, lo); hi2 = None if np.isinf(hi) else hi
        if hi2 is None or hi2 >= lo2: boxes.append((lo2, hi2))
    return boxes
