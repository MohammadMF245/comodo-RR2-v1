"""
Semi-parametric online sparse Gaussian process (generic, model-independent).

One scalar GP per output:

    f_i(xi) = theta_i^T phi_i(xi) + r_i(xi),   r_i ~ GP(0, kappa_i),   theta_i ~ N(mu_th0, Sig_th0)

    phi_i(xi)  : [1, tanh(xi), exp(-||xi - c_j||^2 / l_j)_{j=1..m}]   bounded basis: linear + RBF network
    kappa_i    : A exp(-1/2 ||xi - xi'||^2_{W^-1}) + sigma^2 delta_ab

Recursive posterior, online update, sparsification, hyper-parameters by marginal likelihood.

State of one GP (recursive variables):
    X      (N, d)       feature vectors in the non-parametric data pool
    alpha  (N,)         K^-1 Y
    beta   (N, n_th)    K^-1 Phi^T
    C      (N, N)       K^-1
    mu_th  (n_th,)      parametric posterior mean
    Sig_th (n_th, n_th) parametric posterior covariance

A "bank" groups one GP per output column and is what the controller uses:
    build_spgp_bank -> spgp_bank_predict / spgp_bank_update / make_f_pred_spgp
All arrays are NumPy float64; functions return new states (no hidden globals).
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


# =============================================================================
# Kernel and basis
# =============================================================================
def se_kernel(X1, X2, hyp):
    """A exp(-1/2 sum_k (x1_k - x2_k)^2 / W_k)  (no noise term; noise is added on the diagonal)."""
    W = np.exp(hyp["logW"])
    D = (X1[:, None, :] - X2[None, :, :])**2 / W
    return np.exp(hyp["logA"]) * np.exp(-0.5*D.sum(-1))


def rbf_basis(X, hyp):
    """Basis Phi (n_th, n) = [1; tanh(x) (if hyp['lin']); exp(-||x - c_j||^2 / l_j)]
    (Phi in R^{n_th x |D|})."""
    parts = [np.ones((1, X.shape[0]))]
    if hyp.get("lin", False):
        parts.append(np.tanh(X).T)
    if hyp["centers"].shape[0] > 0:
        d2 = ((X[:, None, :] - hyp["centers"][None, :, :])**2).sum(-1)      # (n, m)
        parts.append(np.exp(-d2/np.exp(hyp["logl"])).T)
    return np.vstack(parts)


def n_theta(hyp):
    return 1 + (hyp["centers"].shape[1] if hyp.get("lin", False) else 0) + hyp["centers"].shape[0]


# =============================================================================
# Batch initialisation
# =============================================================================
def spgp_init(X, y, hyp, jitter=1e-10):
    """Batch posterior on data (X, y) in recursive form."""
    n = X.shape[0]
    sn2 = np.exp(hyp["logsn2"])
    K = se_kernel(X, X, hyp) + (sn2 + jitter)*np.eye(n)
    cf = cho_factor(K, lower=True)
    C = cho_solve(cf, np.eye(n))
    Phi = rbf_basis(X, hyp)                                   # (n_th, n)
    alpha = C @ y
    beta = C @ Phi.T                                          # (n, n_th)
    Sig0_inv = np.eye(n_theta(hyp))/np.exp(hyp["logsth2"])    # Sig_th0 = s_th^2 I
    Sig_th = np.linalg.inv(Phi @ beta + Sig0_inv)             
    Sig_th = 0.5*(Sig_th + Sig_th.T)
    mu_th = Sig_th @ (Phi @ alpha + Sig0_inv @ hyp["mu_th0"]) 
    return {"X": X.copy(), "alpha": alpha, "beta": beta, "C": C, "mu_th": mu_th, "Sig_th": Sig_th}


# =============================================================================
# Prediction
# =============================================================================
def spgp_predict(state, hyp, Xs):
    """Predictive mean and variance (of a noisy observation) at Xs (n*, d)."""
    Ks = se_kernel(state["X"], Xs, hyp)                       # (N, n*)
    Phis = rbf_basis(Xs, hyp)                                 # (n_th, n*)
    R = state["beta"].T @ Ks - Phis                           # (n_th, n*)
    mean = Phis.T @ state["mu_th"] + (state["alpha"] - state["beta"] @ state["mu_th"]) @ Ks
    kss = np.exp(hyp["logA"]) + np.exp(hyp["logsn2"])
    var = (np.einsum("in,ij,jn->n", R, state["Sig_th"], R) + kss
           - np.einsum("in,ij,jn->n", Ks, state["C"], Ks))
    return mean, np.maximum(var, 1e-12)


# =============================================================================
# Online update
# =============================================================================
def spgp_update(state, hyp, xi, y_new):
    """Add one observation (xi (d,), y_new) to the recursive representation."""
    X, alpha, beta, C = state["X"], state["alpha"], state["beta"], state["C"]
    mu_th, Sig_th = state["mu_th"], state["Sig_th"]
    xi = np.atleast_2d(xi)
    ks = se_kernel(X, xi, hyp)[:, 0]                          # K*(xi_t)       (N,)
    kss = np.exp(hyp["logA"]) + np.exp(hyp["logsn2"])         # K**(xi_t)
    phi = rbf_basis(xi, hyp)[:, 0]                            # Phi*(xi_t)     (n_th,)
    Cks = C @ ks
    gamma = kss - ks @ Cks                                    # 1/b = predictive variance of r at xi_t
    if not gamma > 0.5*np.exp(hyp["logsn2"]):                 # numerically redundant point: skip
        return state
    b = 1.0/gamma                          
    a = b*(alpha @ ks - y_new)                               
    S = np.r_[Cks, -1.0]                                      # T[C K*] - e
    R = beta.T @ ks - phi                                     # R_t(xi_t)      (n_th,)
    mu_gp = phi @ mu_th + (alpha - beta @ mu_th) @ ks         # mu^GP_t(xi_t)
    S_gp = R @ Sig_th @ R + 1.0/b                             # Sigma^GP_t(xi_t)
    Lam = Sig_th @ R / S_gp                             

    N = alpha.size
    C_new = np.zeros((N + 1, N + 1)); C_new[:N, :N] = C
    new = {"X": np.vstack([X, xi]),
           "alpha": np.r_[alpha, 0.0] + a*S,                                
           "beta": np.vstack([beta, np.zeros(beta.shape[1])]) + b*np.outer(S, R),  
           "C": C_new + b*np.outer(S, S),                                   
           "mu_th": mu_th + Lam*(mu_gp - y_new),                            
           "Sig_th": (np.eye(mu_th.size) - np.outer(Lam, R)) @ Sig_th}     
    new["Sig_th"] = 0.5*(new["Sig_th"] + new["Sig_th"].T)
    new["C"] = 0.5*(new["C"] + new["C"].T)
    return new


# =============================================================================
# Sparsification
# =============================================================================
def spgp_scores(state):
    """Elimination score (C.1): |alpha - beta mu_th|_j / C_jj."""
    return np.abs(state["alpha"] - state["beta"] @ state["mu_th"]) / np.diag(state["C"])


def spgp_delete(state, m):
    """Remove pool point m from the non-parametric part; mu_th, Sig_th are kept."""
    C, alpha, beta = state["C"], state["alpha"], state["beta"]
    keep = np.arange(alpha.size) != m
    c_m = C[m, keep]                                          # T_m^-[V_m[C]]
    Cmm = C[m, m]
    return {"X": state["X"][keep],
            "alpha": alpha[keep] - c_m*alpha[m]/Cmm,                          
            "beta": beta[keep] - np.outer(c_m, beta[m])/Cmm,            
            "C": C[np.ix_(keep, keep)] - np.outer(c_m, c_m)/Cmm,          
            "mu_th": state["mu_th"], "Sig_th": state["Sig_th"]}


def spgp_sparsify(state, n_max):
    """Delete lowest-score points until the pool has n_max points."""
    while state["alpha"].size > n_max:
        state = spgp_delete(state, int(np.argmin(spgp_scores(state))))
    return state


# =============================================================================
# Hyper-parameters
#   1) basis (RBF centres, width) pre-trained by ridge regression -> prior mean mu_th0
#   2) kernel (A, W, sigma^2) and prior variance s_th^2 by the marginal likelihood
#      y ~ N(Phi^T mu_th0, K + Phi^T Sig_th0 Phi)         
# =============================================================================
def pretrain_basis(X, y, n_centers, rng, lin=True, l_grid=(2.0, 5.0, 11.0, 22.0, 44.0, 88.0),
                   lam_grid=(1e-6, 1e-4, 1e-2), val_frac=0.2):
    """RBF-network (+ linear) basis: k-means centres, width and ridge weights by validation."""
    from sklearn.cluster import KMeans
    d = X.shape[1]
    centers = (KMeans(n_clusters=n_centers, n_init=3, random_state=int(rng.integers(1 << 31)))
               .fit(X).cluster_centers_ if n_centers > 0 else np.zeros((0, d)))
    perm = rng.permutation(X.shape[0]); n_val = int(val_frac*X.shape[0])
    va, tr = perm[:n_val], perm[n_val:]
    best = None
    for l in (l_grid if n_centers > 0 else (1.0,)):
        hyp = {"centers": centers, "logl": np.log(np.full(n_centers, l)), "lin": lin}
        Ftr, Fva = rbf_basis(X[tr], hyp).T, rbf_basis(X[va], hyp).T
        for lam in lam_grid:
            w = np.linalg.solve(Ftr.T @ Ftr + lam*np.eye(Ftr.shape[1]), Ftr.T @ y[tr])
            err = np.mean((Fva @ w - y[va])**2)
            if best is None or err < best[0]:
                best = (err, l, lam)
    _, l, lam = best
    hyp = {"centers": centers, "logl": np.log(np.full(n_centers, l)), "lin": lin}
    F = rbf_basis(X, hyp).T
    hyp["mu_th0"] = np.linalg.solve(F.T @ F + lam*np.eye(F.shape[1]), F.T @ y)
    return hyp


def pack_hyp(hyp):
    return np.r_[hyp["logA"], hyp["logW"], hyp["logsn2"], hyp["logsth2"]]


def unpack_hyp(p, template):
    d = template["logW"].size
    return dict(template, logA=p[0], logW=p[1:1 + d], logsn2=p[1 + d], logsth2=p[2 + d])


def neg_log_marginal(p, X, y, template):
    """-log N(y; Phi^T mu_th0, K + Phi^T Sig_th0 Phi)."""
    hyp = unpack_hyp(p, template)
    n = X.shape[0]
    Phi = rbf_basis(X, hyp)
    Kt = (se_kernel(X, X, hyp) + np.exp(hyp["logsn2"])*np.eye(n)
          + np.exp(hyp["logsth2"])*Phi.T @ Phi + 1e-9*np.eye(n))
    try:
        L = np.linalg.cholesky(Kt)
    except np.linalg.LinAlgError:
        return 1e10
    v = np.linalg.solve(L, y - Phi.T @ hyp["mu_th0"])
    return 0.5*v @ v + np.log(np.diag(L)).sum() + 0.5*n*np.log(2*np.pi)


def init_hyp(X, y, n_centers, rng, use_basis=True, lin=True):
    """Pre-trained basis + initial kernel hyper-parameters."""
    d = X.shape[1]
    if use_basis:
        hyp = pretrain_basis(X, y, n_centers, rng, lin=lin)
        res = y - rbf_basis(X, hyp).T @ hyp["mu_th0"]
    else:                                   # "without base": constant only, zero prior mean
        hyp = {"centers": np.zeros((0, d)), "logl": np.zeros(0), "lin": False, "mu_th0": np.zeros(1)}
        res = y
    v = max(np.var(res), 1e-12)
    hyp.update({"logA": np.log(v), "logW": np.log(np.full(d, float(d))),
                "logsn2": np.log(1e-2*v), "logsth2": np.log(1e-2)})
    return hyp


DEFAULT_BOUNDS = {"A": (1e-6, 0.3), "sqrtW": (0.5, 5.0), "sn2": (1e-4, 1.0), "sth2": (1e-6, 1.0)}


def tune_hyp(X, y, hyp0, maxiter=200, verbose=False, bounds=None):
    """Maximise the marginal likelihood over (A, W, sigma^2, s_th^2); basis kept fixed.

    Bounds (targets are scaled to unit variance) keep the GP a LOCAL residual model:
    small amplitude A, short length scales sqrt(W), a noise floor. Without them the
    likelihood prefers a large, smooth GP that competes with the basis; the parametric
    posterior then degrades and the model collapses after sparsification."""
    b = dict(DEFAULT_BOUNDS, **(bounds or {}))
    p0 = pack_hyp(hyp0)
    lb = [np.log(b["A"][0])] + [2*np.log(b["sqrtW"][0])]*hyp0["logW"].size + [np.log(b["sn2"][0]), np.log(b["sth2"][0])]
    ub = [np.log(b["A"][1])] + [2*np.log(b["sqrtW"][1])]*hyp0["logW"].size + [np.log(b["sn2"][1]), np.log(b["sth2"][1])]
    p0 = np.clip(p0, lb, ub)
    bounds = list(zip(lb, ub))
    res = minimize(neg_log_marginal, p0, args=(X, y, hyp0), method="L-BFGS-B", bounds=bounds,
                   options=dict(maxiter=maxiter))
    if verbose:
        print(f"   NLL {neg_log_marginal(p0, X, y, hyp0):.1f} -> {res.fun:.1f}  ({res.nit} it)")
    return unpack_hyp(res.x, hyp0)


# =============================================================================
# Bank of GPs for the residual outputs
# =============================================================================
def build_spgp_bank(X_norm, Y, out_idx, n_pool, n_centers=64, n_hyp=300, rng=None,
                    use_basis=True, lin=True, maxiter=200, verbose=True, bounds=None,
                    drop_features=()):
    """One semi-parametric sparse GP per output column in out_idx.

    X_norm : (M, d) normalised features (same normalizer as used at run time)
    Y      : (M, n_out_total) residual targets (all columns; only out_idx are learned)
    Steps per output: scale target, pre-train the basis (ridge, all M points) -> mu_th0, tune the
    kernel hyper-parameters on n_hyp points, absorb ALL M points (the parametric posterior sees
    everything), sparsify the non-parametric pool to n_pool.
    drop_features: feature columns ignored by the GP (set to 0). Use it for the steering delta:
    the controller can only feed the PREVIOUS input, and a residual model that depends on delta
    closes a u -> F_hat -> u loop that can destabilise the steering.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    mask = np.ones(X_norm.shape[1]); mask[list(drop_features)] = 0.0
    X_norm = X_norm*mask
    bank = {"out_idx": list(out_idx), "n_out": Y.shape[1], "n_pool": n_pool, "mask": mask,
            "states": [], "hyps": [], "scales": []}
    for i in out_idx:
        y = Y[:, i]
        s = max(np.std(y), 1e-9)
        sub = rng.choice(X_norm.shape[0], size=min(n_hyp, X_norm.shape[0]), replace=False)
        hyp0 = init_hyp(X_norm, y/s, n_centers, rng, use_basis, lin)
        if verbose:
            print(f"output {i}: tuning hyper-parameters on {sub.size} points")
        hyp = tune_hyp(X_norm[sub], y[sub]/s, hyp0, maxiter=maxiter, verbose=verbose, bounds=bounds)
        state = spgp_sparsify(spgp_init(X_norm, y/s, hyp), n_pool)
        bank["states"].append(state); bank["hyps"].append(hyp); bank["scales"].append(s)
    return bank


def spgp_bank_predict(bank, x_feature):
    """mu (n_out_total,), var (n_out_total,) at one normalised feature vector."""
    mu = np.zeros(bank["n_out"]); var = np.zeros(bank["n_out"])
    xs = np.atleast_2d(np.asarray(x_feature, dtype=float))*bank["mask"]
    for i, st, hyp, s in zip(bank["out_idx"], bank["states"], bank["hyps"], bank["scales"]):
        m, v = spgp_predict(st, hyp, xs)
        mu[i], var[i] = s*m[0], s*s*v[0]
    return mu, var


def spgp_bank_update(bank, x_feature, y_vec, forget=1.0):
    """Online update with one new observation (all learned outputs) + keep the pool size fixed.
    forget < 1 inflates the parametric covariance before every update (Sigma_theta / forget, as in recursive least
    squares with a forgetting factor): the parametric part then keeps adapting to new offsets."""
    xi = np.asarray(x_feature, dtype=float)*bank["mask"]
    for k, (i, hyp, s) in enumerate(zip(bank["out_idx"], bank["hyps"], bank["scales"])):
        st0 = bank["states"][k]
        if forget < 1.0:
            st0 = dict(st0, Sig_th=st0["Sig_th"]/forget)
        st = spgp_update(st0, hyp, xi, float(y_vec[i])/s)
        bank["states"][k] = spgp_sparsify(st, bank["n_pool"])
    return bank


def make_f_pred_spgp(bank):
    """Predictor with the same interface as f_pred_jit: f(x_feature=...) -> (mu, var)."""
    def f_pred(x_feature):
        return spgp_bank_predict(bank, x_feature)
    return f_pred


def spgp_bank_trace(bank):
    """Trace of the parametric posterior covariance Sigma_theta of every GP in the bank (reliability monitoring)."""
    return np.array([np.trace(st["Sig_th"]) for st in bank["states"]])


def derive_pool(bank, n_pool):
    """Bank with a smaller sparse pool: continuing the greedy deletion from the stored pool gives exactly the
    bank that would have been built directly with n_pool."""
    import copy as _copy
    b = _copy.deepcopy(bank)
    b["states"] = [spgp_sparsify(st, n_pool) for st in b["states"]]
    b["n_pool"] = n_pool
    return b
