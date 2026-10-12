"""
Residual compensation F_hat for the backstepping controller.

The GPs are trained in LF (physical) coordinates:  x_LF' = A x_LF + B1 u + B2 d_r + B3 d + F_x,
F_x ~ GP(x_HF, delta, F_xf, d_r). The controller needs it in z-coordinates:  F_hat_z = T @ F_hat_x.
delta is only known AFTER F_hat has been evaluated, so the GP is fed the control input of the
PREVIOUS step (one-step delay, zero-order hold).

An estimator has the signature
    F_hat_fn(t, x_HF, gp_in, T, gp) -> (F_hat_z (6,), info dict)
"""
import numpy as np

from .residual_learning import gp_features


def gp_std_from_cov(cov, n_out):
    """Predictive standard deviation per output from whatever covariance the GP returns
    ((n,), (n, n), (n, 1, 1), ...)."""
    cov = np.asarray(cov, dtype=float)
    if cov.ndim == 2 and cov.shape == (n_out, n_out):
        var = np.diag(cov)
    else:
        var = cov.reshape(n_out, -1)[:, 0]
    return np.sqrt(np.maximum(var, 0.0))


def F_hat_zero(t, x_HF, gp_in, T, gp):
    """No residual model: F_hat = 0 (z-coordinates)."""
    return np.zeros(6), {"mu": np.zeros(7), "std": np.zeros(7), "F_hat_lf": np.zeros(6),
                         "F_hat_v": 0.0, "x_feature": None}


def F_hat_gp(t, x_HF, gp_in, T, gp):
    """GP residual estimate mapped to z-coordinates.

    gp_in : dict(delta=<control input of the PREVIOUS step>, F_xf, d_roll, kappa)
    gp    : dict(f_pred       = predictor f_pred(x_feature=...) -> (mu (7,), cov),
                 normalizer   = training-feature normalizer,
                 rows_used    = (6,) 0/1 mask on the x_LF residual rows fed to the controller,
                 max_std      = None or (6,)/float: rows whose predictive std exceeds it are set to 0,
                 gain         = scalar in [0, 1] scaling the compensation,
                 ood_limit    = None or float: GP off when a normalised (unmasked) feature exceeds it,
                 ood_mask     = feature mask for the OOD test,
                 use_vx_residual, gain_vx : learned vx0' residual for the speed controller)
    Returns F_hat_z (6,) and an info dict (mu, std, F_hat_lf, F_hat_v, ood, x_feature)."""
    x_feature = gp_features(x_HF, gp_in["delta"], gp_in["F_xf"], gp_in["d_roll"], gp_in["kappa"],
                            gp["normalizer"])
    mu, cov = gp["f_pred"](x_feature=x_feature)
    mu = np.asarray(mu, dtype=float).ravel()
    std = gp_std_from_cov(cov, mu.size)

    F_hat_lf = gp["gain"]*mu[:6]*np.asarray(gp["rows_used"], dtype=float)   # residual of x_LF'
    if gp.get("max_std") is not None:
        F_hat_lf = np.where(std[:6] <= gp["max_std"], F_hat_lf, 0.0)
    # reliability guard: outside the support of the training data -> no compensation
    ood = False
    if gp.get("ood_limit") is not None:
        ood = bool(np.max(np.abs(np.asarray(x_feature))*np.asarray(gp.get("ood_mask", 1.0))) > gp["ood_limit"])
        if ood:
            F_hat_lf = np.zeros(6)
    F_hat_z = T @ F_hat_lf                                                   # x -> z domain

    # residual of the longitudinal sub-dynamics vx0' (GP output 6): goes to the speed controller
    F_hat_v = 0.0
    if gp.get("use_vx_residual", False) and mu.size > 6:
        F_hat_v = gp.get("gain_vx", gp["gain"])*mu[6]
        if gp.get("max_std") is not None and std[6] > gp["max_std"]:
            F_hat_v = 0.0
    return F_hat_z, {"mu": mu, "std": std, "F_hat_lf": F_hat_lf, "F_hat_v": 0.0 if ood else F_hat_v,
                     "ood": ood, "x_feature": x_feature}


def F_hat_from_lf_residual(F_hat_lf, T):
    """Residual of x_LF' (LF physical coordinates) -> z-coordinates:  F_hat_z = T @ F_hat_lf."""
    return T @ np.asarray(F_hat_lf, dtype=float)


def new_gp_memory(u_prev0=0.0):
    """Mutable memory shared by the simulation loop and the RHS:
       u_prev   : control input of the previous step
       u_gp     : (optionally low-pass filtered) previous input actually fed to the GP
       F_hat    : F_hat_z held constant during one integration step
       F_hat_lf : filtered F_hat in x_LF coordinates
       F_hat_v  : filtered residual of vx0' (longitudinal controller)
       F_xf_prev: traction force applied in the previous step
       info     : last GP output (for logging)."""
    return {"u_prev": float(u_prev0), "u_gp": float(u_prev0), "F_hat": np.zeros(6),
            "F_hat_lf": None, "info": None, "F_hat_v": None, "F_xf_prev": None}


def lowpass(y_old, x_new, dt, tau):
    """Discrete first-order low-pass (tau = 0: no filtering)."""
    return x_new if tau <= 0.0 else y_old + min(dt/tau, 1.0)*(x_new - y_old)
