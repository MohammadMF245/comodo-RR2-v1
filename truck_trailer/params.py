"""
Vehicle parameters of the tractor-semitrailer and helpers to build perturbed plant parameters.

The same dict is used by every model in the package (LF design model, HF plant, controller).
"""
import copy

DEFAULT_PARAMS = {
    # -------------------------
    # Tractor geometry [m]
    # -------------------------
    "lf": 1.11,      # CG to front axle
    "lr": 2.79,      # CG to rear axle
    "lh": 3.39,      # CG to hitch

    # -------------------------
    # Trailer geometry [m]
    # -------------------------
    "l10": 5.2,      # hitch to trailer CG
    "l11": 4.8,      # trailer CG to axle
    "l1": 10.0,      # hitch to trailer axle

    # -------------------------
    # Mass properties
    # -------------------------
    "M0": 5760.0,    # tractor mass [kg]
    "M1": 6665.0,    # trailer mass [kg]
    "Iz0": 35137.6,  # tractor yaw inertia [kg m^2]
    "Iz1": 54255.0,  # trailer yaw inertia [kg m^2]

    # -------------------------
    # Tire cornering stiffness [N/rad]
    # -------------------------
    "Cf": 259752,    # front axle
    "Cr": 259752,    # rear axle
    "C1": 259752,    # trailer axle

    # -------------------------
    # Resistance force
    # -------------------------
    "Cr0": 0.01,          # 0th order
    "Cr1": 2*10**(-4),    # 2nd order

    # -------------------------
    # Tire-road friction
    # -------------------------
    "mu": 0.9,

    # -------------------------
    # Static vertical load [N]
    # -------------------------
    "Fz": 35000.0,

    # -------------------------
    # Pacejka tire model
    # -------------------------
    "C_pacejka": 1.3,
    "E_pacejka": -1.6,

    # -------------------------
    # Speed regularisation [m/s]
    # -------------------------
    "v_eps": 0.0002,

    "g": 9.81,
}


def default_params(**overrides):
    """A fresh copy of DEFAULT_PARAMS, optionally with some entries replaced."""
    p = copy.deepcopy(DEFAULT_PARAMS)
    p.update(overrides)
    return p


def make_hf_params(params_nom, scale=None, override=None):
    """Plant (HF) parameters from the nominal ones.

    scale    : {key: factor}  multiplicative uncertainty, e.g. {"M1": 1.7}
    override : {key: value}   absolute replacement
    The design model / controller keep using params_nom.
    """
    p = params_nom.copy()
    for k, f in (scale or {}).items():
        p[k] = p[k]*f
    for k, v in (override or {}).items():
        p[k] = v
    return p
