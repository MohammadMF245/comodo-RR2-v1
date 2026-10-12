"""
Lateral tire models  Fy = f(alpha, C, p).

Every model has the signature  model(alpha, C, p) -> Fy  where
    alpha : slip angle [rad] (scalar or array)
    C     : cornering stiffness [N/rad]
    p     : parameter dict (uses 'mu', 'Fz' and, for Pacejka, 'C_pacejka', 'E_pacejka')
Sign convention: Fy = -C*alpha for small slip.

New models can be registered with `register_tire_model(name, fn)` and are then available
to TruckTrailerDynamics(tire_model=name).
"""
import numpy as np


def linear_tire(alpha, C, p):
    return -C*alpha


def saturated_linear_tire(alpha, C, p):
    Fy = -C*alpha
    Fy_max = p["mu"]*p["Fz"]
    return np.clip(Fy, -Fy_max, Fy_max)


def fiala_tire(alpha, C, p):
    mu, Fz = p["mu"], p["Fz"]
    t = np.tan(alpha)
    alpha_sl = np.arctan(3*mu*Fz/C)
    if np.ndim(alpha) == 0:
        if abs(alpha) < alpha_sl:
            return (-C*t
                    + (C**2/(3*mu*Fz))*abs(t)*t
                    - (C**3/(27*mu**2*Fz**2))*t**3)
        return -mu*Fz*np.sign(alpha)
    poly = (-C*t + (C**2/(3*mu*Fz))*np.abs(t)*t - (C**3/(27*mu**2*Fz**2))*t**3)
    return np.where(np.abs(alpha) < alpha_sl, poly, -mu*Fz*np.sign(alpha))


def pacejka_tire(alpha, C, p):
    D = p["mu"]*p["Fz"]
    B = -C/(p["C_pacejka"]*D)
    return D*np.sin(p["C_pacejka"]*np.arctan(B*alpha - p["E_pacejka"]*(B*alpha - np.arctan(B*alpha))))


TIRE_MODELS = {
    "linear": linear_tire,
    "sat_linear": saturated_linear_tire,
    "fiala": fiala_tire,
    "pacejka": pacejka_tire,
}


def register_tire_model(name, fn):
    """Make a custom tire model fn(alpha, C, p) available by name."""
    TIRE_MODELS[name.lower()] = fn


def get_tire_model(name_or_fn):
    """Tire model by name (see TIRE_MODELS) or a callable fn(alpha, C, p)."""
    if callable(name_or_fn):
        return name_or_fn
    try:
        return TIRE_MODELS[name_or_fn.lower()]
    except KeyError:
        raise ValueError(f"Unknown tire model '{name_or_fn}'. Available: {sorted(TIRE_MODELS)}") from None
