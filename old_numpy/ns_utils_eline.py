# ns_utils.py

import os
import math

import numpy as np
import matplotlib.pyplot as plt

# Manual caches: coefficient arrays and the constants derived from them.
_CACHE_A = {}
_CACHE_PSI = {}
_CACHE_CONST = {}
# TODO: maybe save these cache in a csv or something so we can access it when we want without having to recompute everything?


# ----------------------------------------------------------------------------
# 1. The canonical degree-(2d+1) family
# ----------------------------------------------------------------------------


def cbin(j):
    """Central binomial coefficient C(2j, j)."""
    return math.comb(2 * j, j)


def coeffs_a(d): 
    # TODO: not clear to me how these coefficient are computed 
    # maybe add a remark in the paper on how to wirte p_d(x) = sum_j a[j] x^(2j+1) in closed form
    """Coefficients a[0..d] of p_d(x) = sum_j a[j] x^(2j+1).

    Shared by the scalar and the matrix readings of the iteration.
    """
    if d < 0:
        raise ValueError("d must be nonnegative")
    if d in _CACHE_A:
        return _CACHE_A[d]
    a = np.zeros(d + 1)
    for j in range(d + 1):
        cj = cbin(j) / 4.0**j
        for i in range(j + 1):  # binomial expansion of (1 - x^2)^j
            a[i] += cj * math.comb(j, i) * (-1) ** i
    _CACHE_A[d] = a
    return a


def T_d(x, d):
    """T_d(x) = sum_j C(2j,j) (1-x^2)^j / 4^j."""
    x = np.asarray(x, dtype=float)
    t = 1.0 - x**2
    out = np.zeros_like(x)
    tp = np.ones_like(x)
    # Summed in t = 1-x^2: on [-1,1] every term is positive, so nothing cancels.
    for j in range(d + 1):
        out = out + (cbin(j) / 4.0**j) * tp
        tp = tp * t
    return out


def p_d(x, d):
    """p_d(x) = x T_d(x). Preferred scalar evaluator: no cancellation near x=1."""
    return np.asarray(x, dtype=float) * T_d(x, d)


def p_coeff(x, a):
    """p(x) = sum_j a[j] x^(2j+1), Horner in x^2."""
    x = np.asarray(x, dtype=float)
    x2 = x * x
    out = np.full_like(x, a[-1])
    for j in range(len(a) - 2, -1, -1):
        out = out * x2 + a[j]
    return out * x


def coeffs_full(d):
    """Ascending monomial coefficients of p_d, even entries zero."""
    a = coeffs_a(d)
    full = np.zeros(2 * d + 2)
    full[1::2] = a
    return full


def coeffs_psi(d):
    """Coefficients of psi_d, where p_d(x) - 1 = (x-1)^(d+1) psi_d(x)."""
    if d in _CACHE_PSI:
        return _CACHE_PSI[d]
    q = coeffs_full(d).copy()
    q[0] -= 1.0 # q = p_d-1
    for _ in range(d + 1):
        q, _ = np.polynomial.polynomial.polydiv(q, np.array([-1.0, 1.0])) # q/(x-1)
    _CACHE_PSI[d] = q
    return q


def psi_d(x, d): 
    return np.polynomial.polynomial.polyval(np.asarray(x, dtype=float), coeffs_psi(d))


def psi_at_one(d):
    """|psi_d(1)| = C(2d+2, d+1) / 2^(d+1), the asymptotic error constant."""
    return cbin(d + 1) / 2.0 ** (d + 1)


def constants(d, n_grid):
    """(mu_d, rho_d, lambda_d): sup of |psi_d| on [0,1], ball radius, growth rate."""
    key = (d, n_grid)
    if key in _CACHE_CONST:
        return _CACHE_CONST[key]
    if d < 1:
        raise ValueError("the constants are defined for d >= 1")
    x = np.linspace(0.0, 1.0, n_grid)
    mu = float(np.max(np.abs(psi_d(x, d)))) #TODO: here, why not differentiate psi and find the roots with eg np.roots to get the maximum ?
    rho = mu ** (-1.0 / d)
    lam = float(T_d(1.0 - rho, d))
    _CACHE_CONST[key] = (mu, rho, lam)
    return _CACHE_CONST[key]


# ----------------------------------------------------------------------------
# 2. Scalar orbits and error tracks
# ----------------------------------------------------------------------------


def orbit_pd(u0, d, n_iters):
    """Orbit u_0..u_K of u_{k+1} = p_d(u_k); shape (n_iters+1, size of u0)."""
    if n_iters < 0:
        raise ValueError("n_iters must be nonnegative")
    u = np.atleast_1d(np.asarray(u0, dtype=float)).astype(float).copy()
    out = np.empty((n_iters + 1, u.size))
    out[0] = u
    for k in range(n_iters):
        u = np.asarray(p_d(u, d), dtype=float)
        out[k + 1] = u
    return out


def log10_err_orbit(u0, d, n_iters, switch=1e-8):
    """log10|1 - u_k| for u_0 in (0,1)."""
    u0 = float(u0)
    if not 0.0 < u0 < 1.0:
        raise ValueError("u0 must lie in the open interval (0,1)")
    if n_iters < 0:
        raise ValueError("n_iters must be nonnegative")
    psi = coeffs_psi(d)
    log_psi_1 = math.log10(psi_at_one(d))

    L = np.empty(n_iters + 1)
    u = u0
    e = 1.0 - u
    L[0] = math.log10(e)
    for k in range(n_iters):
        if e > switch: # TODO: add comments on wat switch is used for
            u_next = float(np.clip(p_d(u, d), 0.0, 1.0)) #TODO: clip is really necessary here ?
            e_next = 1.0 - u_next
            if e_next > switch: # TODO: add comments, not clear what is done here. 
                L[k + 1] = math.log10(e_next)
            else:
                val = abs(np.polynomial.polynomial.polyval(u, psi))
                L[k + 1] = (d + 1) * L[k] + math.log10(val)
            u, e = u_next, e_next
        else:
            L[k + 1] = (d + 1) * L[k] + log_psi_1
    return L


def entry_index(L, d, n_grid):
    """First k with |u_k - 1| <= rho_d; -1 if the ball is never reached."""
    rho = constants(d, n_grid)[1]
    hit = np.nonzero(L <= math.log10(rho))[0]
    if len(hit) == 0:
        return -1
    return int(hit[0])


def burnin_bound(u0, d, n_grid):
    """Prop. burnin: ceil( log((1-rho_d)/u0) / log(lambda_d) ), 0 past the ball."""
    _, rho, lam = constants(d, n_grid)
    if u0 >= 1.0 - rho:
        return 0
    return int(math.ceil(math.log((1.0 - rho) / u0) / math.log(lam)))


# ----------------------------------------------------------------------------
# 3. Iterated sign and the scalar profiles built from it
# ----------------------------------------------------------------------------


def sgn_exact(t):
    """Sign with the convention sgn(0) = 0."""
    return np.sign(np.asarray(t, dtype=float))


def sgn_ns(t, d, n_iters, scale=None):
    """Approximate sign: n_iters steps of p_d on t rescaled into [-1,1].

    The rescaling is free because sgn is invariant under positive scalings,
    while the iteration only converges on [-1,1]; each call scales its own
    argument, as the pre-scaling remarks require.
    """
    t = np.asarray(t, dtype=float)
    if scale is None:
        m = float(np.max(np.abs(t))) if t.size else 0.0
        scale = m if m > 0.0 else 1.0
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    u = np.clip(t / scale, -1.0, 1.0)  # guard round-off outside the invariant interval
    for _ in range(n_iters):
        u = np.asarray(p_d(u, d), dtype=float)
    return u


def make_sgn(d, n_iters):
    """Sign surrogate as a one-argument callable, for the operator forms below."""
    return lambda t: sgn_ns(t, d, n_iters)


def relu(u):
    """ReLU(u) = max(u, 0)."""
    return np.maximum(np.asarray(u, dtype=float), 0.0)


def hinge_mu(x, mu):
    """Hinge at mu, ReLU(x - mu); zero at the origin only for mu >= 0."""
    return relu(np.asarray(x, dtype=float) - mu)


def odd_ext(f):
    """Odd extension x -> sgn(x) f(|x|) of a scalar profile (eq. odd-scalar).

    The result vanishes at the origin whatever f(0) is, which is exactly what
    makes the induced operator well posed on rank-deficient arguments.
    """

    def g(x):
        x = np.asarray(x, dtype=float)
        return sgn_exact(x) * np.asarray(f(np.abs(x)), dtype=float)

    return g


def clip_ab(x, alpha, beta):
    """Exact clipping onto [alpha, beta]."""
    if not alpha < beta:
        raise ValueError("require alpha < beta")
    return np.clip(np.asarray(x, dtype=float), alpha, beta)


def clip_sgn(x, alpha, beta, sgn):
    """Clipping as an affine combination of two signs (Lem. scalar-clip)."""
    if not alpha < beta:
        raise ValueError("require alpha < beta")
    x = np.asarray(x, dtype=float)
    ta = alpha - x
    tb = beta - x
    return 0.5 * (alpha + beta + ta * sgn(ta) - tb * sgn(tb))


def soft_g(x, gamma):
    """Exact soft-thresholding at level gamma."""
    if gamma <= 0.0:
        raise ValueError("require gamma > 0")
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.maximum(np.abs(x) - gamma, 0.0)


def soft_sgn(x, gamma, sgn):
    """Soft-thresholding as a combination of signs (Lem. scalar-soft)."""
    if gamma <= 0.0:
        raise ValueError("require gamma > 0")
    x = np.asarray(x, dtype=float)
    tm = x - gamma
    tp = x + gamma
    return 0.5 * (2.0 * x + tm * sgn(tm) - tp * sgn(tp))


class Spline(object):
    """Linear spline on a partition, with its canonical extension.

    Attributes: knots, vals, c (cell slopes), w (slope jumps), eta, theta, kappa.
    """

    def __init__(self, knots, vals):
        self.knots = np.asarray(knots, dtype=float).ravel()
        self.vals = np.asarray(vals, dtype=float).ravel()
        if self.knots.size < 2:
            raise ValueError("need at least two knots")
        if self.knots.shape != self.vals.shape:
            raise ValueError("knots and vals must have the same length")
        if np.any(np.diff(self.knots) <= 0.0):
            raise ValueError("knots must be strictly increasing")
        h = np.diff(self.knots)
        self.c = np.diff(self.vals) / h
        self.w = np.diff(self.c)
        self.interior = self.knots[1:-1]
        self.eta = float(self.vals[0] - self.c[0] * self.knots[0])
        self.theta = float(0.5 * (self.c[0] + self.c[-1]))
        self.kappa = float(self.eta - 0.5 * np.sum(self.w * self.interior))

    def ext(self, x):
        """ReLU representation, the reference value."""
        x = np.asarray(x, dtype=float)
        out = self.eta + self.c[0] * x
        for i in range(len(self.w)):
            out = out + self.w[i] * np.maximum(x - self.interior[i], 0.0) # TODO: maybe define def relu(x): return np.maximum(x, 0.0) ?
        return out

    def ext_sgn(self, x, sgn):
        """Sign representation, the form that lifts to matrices."""
        x = np.asarray(x, dtype=float)
        out = self.kappa + self.theta * x
        for i in range(len(self.w)):
            t = x - self.interior[i]
            out = out + 0.5 * self.w[i] * t * sgn(t)
        return out

    def odd_ext(self, x):
        x = np.asarray(x, dtype=float)
        return np.sign(x) * self.ext(np.abs(x))

    def odd_ext_sgn(self, x, sgn):
        """Odd extension with the outer sign approximated as well."""
        x = np.asarray(x, dtype=float)
        return sgn(x) * self.ext_sgn(np.abs(x), sgn)


# ----------------------------------------------------------------------------
# 4a. Matrix reading of the same coefficients, pre-scaling norms, orbits
# ----------------------------------------------------------------------------


def iter_matrix(X, a):
    """Odd matrix polynomial Phi(X) = sum_j a[j] (X X^T)^j X."""
    X = np.asarray(X, dtype=float)
    G = X @ X.T
    out = a[0] * X
    Pk = X
    for j in range(1, len(a)):
        Pk = G @ Pk  # (X X^T)^j X, built by one product per degree
        out = out + a[j] * Pk
    return out


def spec_norm_svd(M):
    """Exact largest singular value."""
    M = np.asarray(M, dtype=float)
    if M.size == 0:
        return 0.0
    return float(np.linalg.norm(M, 2))


def spec_norm_power(M, n_iters=200, tol=1e-12, seed=0, pad=1.0 + 1e-6):
    """Largest singular value by power iteration on M^T M, inflated by `pad`."""
    M = np.asarray(M, dtype=float)
    if M.size == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(M.shape[1])
    nv = float(np.linalg.norm(v))
    if nv == 0.0:
        return 0.0
    v = v / nv
    s = 0.0
    s_prev = -1.0
    for _ in range(max(1, int(n_iters))):
        w = M @ v
        s = float(np.linalg.norm(w))
        if s == 0.0:
            return 0.0
        z = M.T @ w
        nz = float(np.linalg.norm(z))
        if nz == 0.0:  # v already spans the dominant right singular direction
            break
        v = z / nz
        if abs(s - s_prev) <= tol * max(s, 1.0):
            break
        s_prev = s
    return float(pad * s)


def spec_norm(M, method="svd", **kw):
    """Pre-scaling norm; `method` is 'svd' (exact) or 'power' (matrix-free)."""
    if method == "svd":
        return spec_norm_svd(M)
    if method == "power":
        return spec_norm_power(M, **kw)
    raise ValueError("unknown norm method: %r" % (method,))


def sgn_matrix(M, a, n_iters, scale=None, norm_method="svd", norm_kw=None):
    """Approximate matrix sign: n_iters steps of Phi on M pre-scaled to unit norm.

    Sgn(M/beta) = Sgn(M) for beta > 0, so the pre-scaling that puts the spectrum
    inside the convergence interval [-1,1] leaves the target unchanged.
    """
    M = np.asarray(M, dtype=float)
    if scale is None:
        scale = spec_norm(M, method=norm_method, **(norm_kw or {}))
    if scale <= 0.0:
        return np.zeros_like(M)
    X = M / scale
    for _ in range(n_iters):
        X = iter_matrix(X, a)
    return X


def orbit_matrix(M, a, n_iters, scale=None, norm_method="svd", norm_kw=None):
    """Iterates X_0,...,X_K of eq. ns-recursion, started at the pre-scaled M.

    The whole orbit shares one pre-scaling, so it may be compared against a
    single target: Sgn(M/beta) = Sgn(M) by Lem. prescaling.
    """
    M = np.asarray(M, dtype=float)
    if n_iters < 0:
        raise ValueError("n_iters must be nonnegative")
    if scale is None:
        scale = spec_norm(M, method=norm_method, **(norm_kw or {}))
    if scale <= 0.0:
        return [np.zeros_like(M) for _ in range(n_iters + 1)]
    X = M / scale
    out = [X]
    for _ in range(n_iters):
        X = iter_matrix(X, a)
        out.append(X)
    return out


def make_sgn_matrix(d, n_iters, norm_method="svd", norm_kw=None):
    """Sgn surrogate as a one-argument callable on matrices.

    Every call pre-scales its own argument, which is what the pre-scaling remarks
    of Sec. clip and Sec. prox require of the inner evaluations.
    """
    a = coeffs_a(d)

    def Sgn(M):
        return sgn_matrix(M, a, n_iters, norm_method=norm_method, norm_kw=norm_kw)

    return Sgn


# ----------------------------------------------------------------------------
# 4b. Random matrices and the shared configuration front end
# ----------------------------------------------------------------------------


def rand_orth(q, k, rng):
    """Semi-orthogonal q x k factor, q >= k, from a QR of a Gaussian block."""
    if k > q:
        raise ValueError("need q >= k for a semi-orthogonal q x k factor")
    Q, R = np.linalg.qr(rng.standard_normal((q, k)))
    s = np.sign(np.diag(R))
    s = np.where(s == 0.0, 1.0, s)  # pin the QR sign convention, so runs reproduce
    return Q * s


def rand_gauss(m, n, rng):
    """Plain Gaussian matrix; almost surely full rank, spectrum unprescribed."""
    return rng.standard_normal((m, n))


def rand_gauss_edit(m, n, rng, n_zero=0, n_small=0, s_small=1e-6, unit=True):
    """Gaussian matrix with its smallest singular values overwritten."""
    U, sig, V = calc_svd(rand_gauss(m, n, rng))
    sig = np.array(sig, dtype=float, copy=True)
    r = sig.size
    n_zero, n_small = int(n_zero), int(n_small)
    if n_zero < 0 or n_small < 0 or n_zero + n_small > r:
        raise ValueError("require 0 <= n_zero + n_small <= min(m,n)")
    if s_small < 0.0:
        raise ValueError("s_small must be nonnegative")
    if unit and r and sig[0] > 0.0:
        sig = sig / sig[0]
    if n_small:  # sig is descending, so the tail is the end of the array
        sig[r - n_zero - n_small:r - n_zero] = s_small
    if n_zero:
        sig[r - n_zero:] = 0.0
    return (U * sig) @ V.T


def make_M(cfg):
    """Test matrix of an experiment."""
    rng = np.random.default_rng(cfg["seed"])
    m, n = int(cfg["m"]), int(cfg["n"])
    gen = cfg.get("gen", "gauss")
    if gen == "gauss":
        M = rand_gauss(m, n, rng)
    elif gen == "edit":
        M = rand_gauss_edit(m, n, rng, n_zero=cfg.get("n_zero", 0),
                            n_small=cfg.get("n_small", 0),
                            s_small=cfg.get("s_small", 1e-6),
                            unit=cfg.get("normalize", True))
    else:
        raise ValueError("unknown generator: %r" % (gen,))
    if cfg.get("normalize", True):
        beta = spec_norm_svd(M)  # data normalization, kept exact by design
        if beta > 0.0:
            M = M / beta
    return M


def norm_kwargs(cfg):
    """Keyword block for the pre-scaling norm; empty unless power iteration is asked."""
    if cfg.get("norm_method", "svd") != "power":
        return {}
    return {"n_iters": cfg.get("norm_power_iters", 300),
            "tol": cfg.get("norm_tol", 1e-12),
            "seed": cfg.get("seed", 0),
            "pad": cfg.get("norm_pad", 1.0 + 1e-6)}


def pre_norm(M, cfg):
    """The scalar beta used to pre-scale M, as configured."""
    return spec_norm(M, method=cfg.get("norm_method", "svd"), **norm_kwargs(cfg))


def sgn_cfg(cfg, k):
    """Sgn surrogate with k iterations, at the degree and the norm of this run."""
    return make_sgn_matrix(cfg["d"], int(k), norm_method=cfg.get("norm_method", "svd"),
                           norm_kw=norm_kwargs(cfg))


# ----------------------------------------------------------------------------
# 4c. Reference layer: the operators as read off an SVD
# ----------------------------------------------------------------------------


def calc_svd(M):
    """Thin SVD (U, sigma, V), sigma descending, in the sense of Def. thin-svd."""
    U, sig, Vt = np.linalg.svd(np.asarray(M, dtype=float), full_matrices=False)
    return U, sig, Vt.T


def svdvals(M):
    return np.linalg.svd(np.asarray(M, dtype=float), compute_uv=False)


def rank_tol(M, sig): # TODO: not clear to me what is the purpose of this function, maybe add a comment / equation number of the report
    """Numerical-rank threshold, the float reading of the convention sgn(0) = 0."""
    sig = np.asarray(sig, dtype=float)
    if sig.size == 0:
        return 0.0
    return float(max(np.shape(M)) * np.finfo(float).eps * sig[0])


def op_svd(M, f):
    """Reference spectral operator U diag(f(sigma)) V^T of eq. svd-op.

    `f` is evaluated on the singular values only, so it should be handed in as an
    odd extension whenever the argument may be rank deficient.
    """
    U, sig, V = calc_svd(M)
    return (U * np.asarray(f(sig), dtype=float)) @ V.T


def sgn_svd(M, tol=None):
    """Exact Sgn(M) of Def. sgn, singular values thresholded at `tol`."""
    U, sig, V = calc_svd(M)
    if tol is None:
        tol = rank_tol(M, sig)
    return (U * (sig > tol).astype(float)) @ V.T


def spec_coords(Y, U, V): #TODO: not clear what is done here, aren't we suppose to look at U^T diag(Y) V instead of diag(U^T Y V) ?  when do we need diag(U^T Y V)?
    """diag(U^T Y V): the spectral coordinates of Y in the frame (U, V).

    Every map studied here keeps the frame of its argument, so these are the
    numbers the theory predicts; they coincide with the singular values of Y
    whenever the induced scalar profile is nondecreasing on [0,1].
    """
    return np.einsum("ji,ji->i", U, np.asarray(Y, dtype=float) @ V)


def frame_resid(Y, U, V):
    """||Y - U diag(diag(U^T Y V)) V^T||_F: zero iff Y lives in the frame (U,V)."""
    Y = np.asarray(Y, dtype=float)
    c = spec_coords(Y, U, V)
    return float(np.linalg.norm(Y - (U * c) @ V.T))


# ----------------------------------------------------------------------------
# 4d. Decomposition-free expressions, driven by a sign callable
# ----------------------------------------------------------------------------


def clip_map(M, alpha, beta, Sgn, N=None):
    """Eq. clip-expr: (1/2)[(a+b)I + M_a Sgn(M_a)^T - M_b Sgn(M_b)^T] Sgn(M)."""
    if not alpha < beta:
        raise ValueError("require alpha < beta")
    M = np.asarray(M, dtype=float)
    if N is None:
        N = Sgn(M)
    Ma = alpha * N - M
    Mb = beta * N - M
    core = (alpha + beta) * np.eye(M.shape[0]) + Ma @ Sgn(Ma).T - Mb @ Sgn(Mb).T
    return 0.5 * (core @ N)


def hinge_map(M, mu, Sgn, N=None):
    """Eq. mhinge: (1/2)[M_mu Sgn(M_mu)^T Sgn(M) - M_mu], M_mu = mu Sgn(M) - M."""
    M = np.asarray(M, dtype=float)
    if N is None:
        N = Sgn(M)
    Mmu = mu * N - M
    return 0.5 * (Mmu @ Sgn(Mmu).T @ N - Mmu)


def hinge_map_neg(M, mu, Sgn, N=None):
    """Closed form M - mu Sgn(M), valid for mu <= 0 (Prop. mhinge-neg)."""
    if mu > 0.0:
        raise ValueError("the closed form is valid for mu <= 0 only")
    M = np.asarray(M, dtype=float)
    if N is None:
        N = Sgn(M)
    return M - mu * N


def soft_map(M, gamma, Sgn, N=None):
    """Eq. soft-expr; the matrix hinge at a positive threshold (Cor. mh-soft)."""
    if gamma <= 0.0:
        raise ValueError("require gamma > 0")
    return hinge_map(M, gamma, Sgn, N)


def spline_map(M, spl, Sgn, N=None):
    """Eq. spline-expr, hinge form: eta Sgn(M) + c_0 M + sum_i w_i H_{x_i}(M)."""
    M = np.asarray(M, dtype=float)
    if N is None:
        N = Sgn(M)
    out = spl.eta * N + spl.c[0] * M
    for wi, xi in zip(spl.w, spl.interior):
        out = out + wi * hinge_map(M, float(xi), Sgn, N)
    return out


def spline_map_sign(M, spl, Sgn, N=None):
    """Eq. spline-sign-expr, sign form; an independent route to spline_map."""
    M = np.asarray(M, dtype=float)
    if N is None:
        N = Sgn(M)
    out = spl.kappa * N + spl.theta * M
    for wi, xi in zip(spl.w, spl.interior):
        Mx = float(xi) * N - M
        out = out + 0.5 * wi * (Mx @ Sgn(Mx).T @ N)
    return out


def sweep_op(op, k_list, Y_ex, U, V):
    """Run a Sgn-based expression at each NS budget k in `k_list`."""
    out = {}
    for k in k_list:
        Y = op(int(k))
        out[int(k)] = {
            "Y": Y,
            "coords": spec_coords(Y, U, V),
            "errF": float(np.linalg.norm(Y - Y_ex)),
            "err2": float(np.linalg.norm(Y - Y_ex, 2)),
            "resid": frame_resid(Y, U, V),
        }
    return out


# ----------------------------------------------------------------------------
# 5. Minimal plotting helpers
# ----------------------------------------------------------------------------


def setup_ax(ax, xlabel, ylabel, title, legend=True):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    if legend:
        ax.legend(fontsize=8)


def plot_spectrum(ax, sig, exact, approx, ks, title, xaxis="index",
                  ylabel="spectral coordinate"):
    """Input spectrum (dotted), target (dashed), and the NS values at a few k."""
    sig = np.asarray(sig, dtype=float)
    order = np.argsort(sig)  # ascending, and robust to an unsorted input
    s = sig[order]
    if xaxis == "index":
        x, xlabel = np.arange(1, s.size + 1), "index i (increasing sigma)"
    elif xaxis == "sigma":
        x, xlabel = s, r"$\sigma_i(M)$"
    else:
        raise ValueError("xaxis must be 'index' or 'sigma'")
    ax.plot(x, s, ":", color="0.35", lw=1.2, label=r"$\sigma_i(M)$")
    for k, v in zip(ks, approx):
        ax.plot(x, np.asarray(v, dtype=float)[order], lw=1.0, label="k=%d" % int(k))
    ax.plot(x, np.asarray(exact, dtype=float)[order], "k--", lw=1.3, label="exact (SVD)")
    setup_ax(ax, xlabel, ylabel, title)


def plot_decay(ax, ks, errs, labels, title, xlabel="NS steps k", ylabel="error",
               floor=1e-18):
    """Semilog decay of one or several error tracks, floored away from zero."""
    for e, lab in zip(errs, labels):
        ax.semilogy(ks, np.maximum(np.asarray(e, dtype=float), floor), "o-",
                    ms=3, lw=1.0, label=lab)
    setup_ax(ax, xlabel, ylabel, title)


def output(fig, outdir, name, dpi, show):
    fig.tight_layout()
    if outdir:
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        fig.savefig(os.path.join(outdir, name), dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)


# ----------------------------------------------------------------------------
# 6. Self-check against the closed forms of Cor. low-degree
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    N = 20001
    x = np.linspace(-1.0, 1.0, 501)
    print("a(1)  =", np.round(coeffs_a(1), 6), "   expected [1.5, -0.5]")
    print("a(2)  =", np.round(coeffs_a(2), 6), "   expected [1.875, -1.25, 0.375]")
    print("psi_1 =", np.round(coeffs_psi(1), 6), "   expected [-1, -0.5]")
    print("psi_2 =", np.round(coeffs_psi(2), 6), "   expected [1, 1.125, 0.375]")
    mu1, rho1, lam1 = constants(1, N)
    mu2, rho2, lam2 = constants(2, N)
    print("d=1: mu=%.6f (1.5)  rho=%.6f (0.666667)  lam=%.6f (1.444444)" % (mu1, rho1, lam1))
    print("d=2: mu=%.6f (2.5)  rho=%.6f (0.632456)  lam=%.6f (1.712983)" % (mu2, rho2, lam2))
    for d in range(1, 7):
        a = coeffs_a(d)
        gap = np.max(np.abs(p_d(x, d) - p_coeff(x, a)))
        res = np.max(np.abs(p_d(x, d) - 1.0 - (x - 1.0) ** (d + 1) * psi_d(x, d)))
        print("d=%d: |p_d - p_coeff| = %.3e, deflation residual = %.3e, |psi_d(1)| = %.6f"
              % (d, gap, res, psi_at_one(d)))
    # The two readings of a agree: scalar orbit vs 1x1 matrix orbit.
    M1 = np.array([[0.3]])
    print("scalar vs matrix reading at x=0.3, d=2, 3 steps: %.3e"
          % abs(float(sgn_matrix(M1, coeffs_a(2), 3)[0, 0])
                - float(orbit_pd(0.3, 2, 3)[-1, 0])))

    # Matrix layer, on both generators and both pre-scaling norms.
    cfg = {"m": 8, "n": 6, "seed": 0, "gen": "gauss", "n_zero": 2, "n_small": 1,
           "s_small": 1e-6, "d": 3}
    spl = Spline([0.0, 0.3, 0.7, 1.0], [0.0, 0.2, 0.6, 0.8])
    fc = odd_ext(lambda t: clip_ab(t, -0.2, 0.5))
    for gen in ("gauss", "edit"):
        cfg["gen"] = gen
        Mt = make_M(cfg)
        U, s, V = calc_svd(Mt)
        Sgn = sgn_cfg(cfg, 14)
        Yc = clip_map(Mt, -0.2, 0.5, Sgn)
        print("gen=%-5s rank=%d  svd/power %.9f/%.9f  sgn %.2e  clip %.2e  soft %.2e"
              "  spline forms %.2e  resid %.2e"
              % (gen, int(np.sum(s > rank_tol(Mt, s))), spec_norm(Mt, "svd"),
                 spec_norm(Mt, "power", n_iters=300),
                 np.linalg.norm(Sgn(Mt) - sgn_svd(Mt)),
                 np.linalg.norm(Yc - op_svd(Mt, fc)),
                 np.linalg.norm(soft_map(Mt, 0.2, Sgn)
                                - op_svd(Mt, lambda t: soft_g(t, 0.2))),
                 np.linalg.norm(spline_map(Mt, spl, Sgn)
                                - spline_map_sign(Mt, spl, Sgn)),
                 frame_resid(Yc, U, V)))

