"""
Model residual HF - LF: definition, GP features, training data generation and GP-bank training.

Residual (targets of the GP):
    res_x = d/dt x_LF (HF)  -  (A_big x_LF + B1_big delta + B2_big d_r + B3_big d)      (6,)
    res_v = vx0' (HF)       -  nominal longitudinal model                               (1,)
GP features (11, see GP_FEATURE_NAMES):
    [e, psi_tilde, vy0, r0, phi_dot, phi, vx0, delta, F_xf, d_roll, kappa]
"""
import numpy as np

from .lf_model import hf_to_lf, steady_state_lf
from .spgp import build_spgp_bank

GP_FEATURE_NAMES = ["e", "psi_tilde", "vy0", "r0", "phi_dot", "phi", "vx0", "delta", "F_xf", "d_roll", "kappa"]
GP_TARGET_NAMES = ["res_e", "res_psi_tilde", "res_vy0", "res_r0", "res_phi_dot", "res_phi", "res_vx0"]
FEATURE_INDEX = {n: i for i, n in enumerate(GP_FEATURE_NAMES)}


# =============================================================================
# Residual
# =============================================================================
def residual_hf_to_lf(state_HF, delta, F_xf, d_roll, d0, p, dx_HF, lf):
    """True residual HF - LF of x_LF' and of vx0'.

    state_HF : HF state (10,);  dx_HF: its HF derivative
    delta, F_xf : applied inputs;  d_roll = g sin(phi_roll) included in the LF model;  d0 = kappa*vx0
    p  : (nominal) parameters of the longitudinal model;  lf : LF design model dict
    Returns res_x (6,), res_v (scalar)."""
    state_LF = hf_to_lf(state_HF)
    e, psi_t, X0, Y0, psi0, psi1, vx0, vy0, r0, r1 = state_HF
    vy1_est = vy0 - p['lh']*r0 - p['l10']*r1 - vx0*(psi1 - psi0)
    vx0_dot = (F_xf + p['M0']*vy0*r0 + p['M1']*vy1_est*r1)/(p['M0'] + p['M1'])
    state_LF_dot = (np.asarray(lf['A_big']) @ state_LF
                    + np.asarray(lf['B1_big']).squeeze()*delta
                    + np.asarray(lf['B2_big']).squeeze()*d_roll
                    + np.asarray(lf['B3_big']).squeeze()*d0)
    res_dot_x = hf_to_lf(dx_HF) - state_LF_dot
    res_dot_v = dx_HF[6] - vx0_dot
    return res_dot_x, res_dot_v


resedual_HF_to_LF = residual_hf_to_lf          # name used in the original notebook


# =============================================================================
# Features
# =============================================================================
def gp_features_raw(x_HF, delta, F_xf, d_roll, kappa):
    """Raw GP features from HF state(s) and inputs.

    x_HF : (10,) or (N, 10)  [e, psi_tilde, X0, Y0, psi0, psi1, vx0, vy0, r0, r1]
    delta, F_xf, d_roll, kappa : scalars or (N,) arrays
    returns (11,) or (N, 11) in the order of GP_FEATURE_NAMES
    (X0, Y0, psi0 are dropped: the residual is rotation/translation invariant)."""
    x = np.asarray(x_HF, dtype=float)
    single = x.ndim == 1
    x = np.atleast_2d(x)
    e, psi_t = x[:, 0], x[:, 1]
    psi0, psi1 = x[:, 4], x[:, 5]
    vx0, vy0, r0, r1 = x[:, 6], x[:, 7], x[:, 8], x[:, 9]
    n = x.shape[0]
    F = np.column_stack([
        e, psi_t, vy0, r0,
        r1 - r0,                 # phi_dot
        psi1 - psi0,             # phi (articulation)
        vx0,
        np.broadcast_to(delta, n), np.broadcast_to(F_xf, n),
        np.broadcast_to(d_roll, n), np.broadcast_to(kappa, n),
    ])
    return F[0] if single else F


def make_feature_normalizer(X_data, eps=1e-8):
    """Training statistics (from the UN-normalised X_data)."""
    X_data = np.asarray(X_data, dtype=float)
    return {"mean": X_data.mean(axis=0), "std": X_data.std(axis=0), "eps": eps}


def normalize_features(F_raw, normalizer):
    return (F_raw - normalizer["mean"])/(normalizer["std"] + normalizer["eps"])


def gp_features(x_HF, delta, F_xf, d_roll, kappa, normalizer):
    """Normalised GP input."""
    return normalize_features(gp_features_raw(x_HF, delta, F_xf, d_roll, kappa), normalizer)


# =============================================================================
# Training data around the steady-cornering manifold
# =============================================================================
# Deviations from the steady-cornering manifold, calibrated on closed-loop runs at 10 m/s:
#   DEV_TRACK : steady tracking (after the initial transient)
#   DEV_TRANS : transients like a 0.5 m initial offset / 4 deg bank steps
DEV_TRACK = dict(e=0.1, psi_tilde=0.01, vy0=0.05, r0=0.02, r1=0.02, phi=0.02, delta=0.01)
DEV_TRANS = dict(e=0.5, psi_tilde=0.07, vy0=0.35, r0=0.15, r1=0.05, phi=0.06, delta=0.10)


def sample_operating_points(N, rng, lf_design, V_range, kappa_max_path, a_lat_max, phi_roll_max,
                            dev, a_x_range, params_lf, psir_range=(-np.pi, np.pi), manifold_fn=None, a_x_fn=None):
    """GP dataset inputs, sampled around the steady-cornering manifold.

    lf_design : LF design model the residual is defined against (fixed design speed)
    For each sample:
      vx0 ~ U(V_range);   kappa ~ U(-k_max, k_max), k_max = min(kappa_max_path, a_lat_max/vx0^2)
      phi_roll ~ U(+-phi_roll_max);  d0 = kappa*vx0;  d_roll = g sin(phi_roll)
      x_ss, delta_ss = LF steady state (e = 0) at (vx0, d0, d_roll)   (manifold_fn(vx0) or lf_design)
      state = x_ss + U(+-dev),  delta = delta_ss + U(+-dev['delta'])
      F_xf  = (M0+M1)*a_x + lateral-coupling compensation,  a_x ~ U(a_x_range) (or a_x_fn(vx0, rng))
      psi0 arbitrary (dynamics are rotation invariant); psir = psi0 - psi_tilde
    Returns a list of dicts with state_HF, delta, F_xf, phi_roll, d_roll, d0, kappa, psir, lf."""
    p = params_lf; g = p.get("g", 9.81)
    out = []
    for _ in range(N):
        vx0 = rng.uniform(*V_range)
        lf = lf_design
        lf_man = lf if manifold_fn is None else manifold_fn(vx0)
        k_max = min(kappa_max_path, a_lat_max/vx0**2)
        kappa = rng.uniform(-k_max, k_max)
        phi_roll = rng.uniform(-phi_roll_max, phi_roll_max)
        d0, d_roll = kappa*vx0, g*np.sin(phi_roll)
        x_ss, delta_ss = steady_state_lf(lf_man, d0, d_roll)
        e     = x_ss[0] + rng.uniform(-dev["e"], dev["e"])
        psi_t = x_ss[1] + rng.uniform(-dev["psi_tilde"], dev["psi_tilde"])
        vy0   = x_ss[2] + rng.uniform(-dev["vy0"], dev["vy0"])
        r0    = x_ss[3] + rng.uniform(-dev["r0"], dev["r0"])
        r1    = x_ss[3] + x_ss[4] + rng.uniform(-dev["r1"], dev["r1"])   # trailer yaw rate
        phi   = x_ss[5] + rng.uniform(-dev["phi"], dev["phi"])
        delta = delta_ss + rng.uniform(-dev["delta"], dev["delta"])
        psi0 = rng.uniform(*psir_range)                       # irrelevant for the residual
        state_HF = np.array([e, psi_t, 0.0, 0.0, psi0, psi0 + phi, vx0, vy0, r0, r1])
        vy1_est = vy0 - p["lh"]*r0 - p["l10"]*r1 - vx0*phi
        a_x = rng.uniform(*a_x_range) if a_x_fn is None else a_x_fn(vx0, rng)
        F_xf = (p["M0"] + p["M1"])*a_x - p["M0"]*vy0*r0 - p["M1"]*vy1_est*r1
        out.append(dict(state_HF=state_HF, delta=delta, F_xf=F_xf, phi_roll=phi_roll,
                        d_roll=d_roll, d0=d0, kappa=kappa, psir=psi0 - psi_t, lf=lf))
    return out


def build_residual_dataset(samples, model_HF, params_ctrl):
    """GP inputs X (N, 11) and targets Y (N, 7) = LF residuals (6) + longitudinal residual (1)."""
    X, Y = [], []
    for s in samples:
        dx_HF, _ = model_HF.full_dynamics(s["state_HF"], s["delta"], s["F_xf"], s["phi_roll"],
                                          kappa=s["kappa"], psir=s["psir"])
        res_x, res_v = residual_hf_to_lf(s["state_HF"], s["delta"], s["F_xf"], s["d_roll"], s["d0"],
                                         params_ctrl, dx_HF, s["lf"])
        X.append(gp_features_raw(s["state_HF"], s["delta"], s["F_xf"], s["d_roll"], s["kappa"]))
        Y.append(np.r_[res_x, res_v])
    return np.array(X), np.array(Y)


def generate_residual_dataset(model_HF, lf_design, params, rng, n_track=700, n_trans=300,
                              V_range=(9.5, 10.5), kappa_max_path=0.0131, a_lat_max=3.0,
                              phi_roll_max=np.deg2rad(5.0), a_x_range=(-0.05, 0.05),
                              dev_track=DEV_TRACK, dev_trans=DEV_TRANS):
    """Standard initial dataset: n_track samples near steady tracking + n_trans samples in transients.
    Returns X_raw (N, 11), Y (N, 7)."""
    common = dict(V_range=V_range, kappa_max_path=kappa_max_path, a_lat_max=a_lat_max,
                  phi_roll_max=phi_roll_max, a_x_range=a_x_range, params_lf=params)
    samples = (sample_operating_points(n_track, rng, lf_design, dev=dev_track, **common)
               + sample_operating_points(n_trans, rng, lf_design, dev=dev_trans, **common))
    return build_residual_dataset(samples, model_HF, params)


# =============================================================================
# GP banks
# =============================================================================
DEFAULT_BANK_VARIANTS = {
    # aware  : the GP receives the road-bank input g*sin(phi_roll) known to the controller
    # unaware: the GP does not see it; an unmodelled bank is then learned online as part of the residual
    # the steering delta (7) is masked in both (the controller can only feed the PREVIOUS input; a residual
    # that depends on it closes a u -> F_hat -> u loop)
    "aware": [FEATURE_INDEX["delta"]],
    "unaware": [FEATURE_INDEX["delta"], FEATURE_INDEX["d_roll"]],
}


def train_residual_gp_banks(X_norm, Y, variants=None, out_idx=(0, 1, 2, 3, 4, 6), n_pool=30,
                            n_centers=64, n_hyp=300, verbose=True, **kw):
    """One semi-parametric sparse GP bank per variant {name: masked feature indices}.
    out_idx: residual outputs learned (row 5, phi, is identically zero)."""
    banks = {}
    for name, masked in (variants or DEFAULT_BANK_VARIANTS).items():
        if verbose:
            print(f"--- GP '{name}': masked features {list(masked)}")
        banks[name] = build_spgp_bank(X_norm, Y, out_idx=list(out_idx), n_pool=n_pool, n_centers=n_centers,
                                      n_hyp=n_hyp, drop_features=list(masked), verbose=False, **kw)
    return banks
