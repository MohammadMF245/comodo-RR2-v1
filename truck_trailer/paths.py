"""
Reference paths and Frenet path quantities.

A path is a dict:
  * figure eight :  {"type": "eight", "A_path": ..., "B_path": ...}   (theta = curve parameter)
  * arc-length parametrised paths from `make_path` ('line', 'uturn', 'sturn', 'lanechange'),
    or from `path_from_segments` with any curvature profile  (theta = arc length)
"""
import numpy as np


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x*x*(3.0 - 2.0*x)


def path_from_segments(segments, ds=0.05, path_type="custom"):
    """Arc-length parametrised path from a curvature profile.
    segments: list of ('const', L, k) | ('ramp', L, k0, k1)  (smooth-step transition of the curvature).
    Beyond its end the path is extended as a straight line (kappa = 0), so end the profile with a
    straight segment to avoid a curvature jump (steering spike) when a run outlasts the path."""
    edges = np.cumsum([0.0] + [seg[1] for seg in segments])
    s = np.arange(0.0, edges[-1] + 0.5*ds, ds)
    kap = np.zeros_like(s)
    for seg, a0, a1 in zip(segments, edges[:-1], edges[1:]):
        m = (s >= a0) & (s <= a1)
        if seg[0] == "const":
            kap[m] = seg[2]
        else:
            kap[m] = seg[2] + (seg[3] - seg[2])*_smoothstep((s[m] - a0)/seg[1])
    dsv = np.diff(s)
    psi = np.r_[0.0, np.cumsum(0.5*(kap[1:] + kap[:-1])*dsv)]
    X = np.r_[0.0, np.cumsum(0.5*(np.cos(psi[1:]) + np.cos(psi[:-1]))*dsv)]
    Y = np.r_[0.0, np.cumsum(0.5*(np.sin(psi[1:]) + np.sin(psi[:-1]))*dsv)]
    return {"s": s, "X": X, "Y": Y, "psi": psi, "kappa": kap, "length": float(s[-1]), "type": path_type}


def _lane_change_path(dy=3.5, L=40.0, L_in=60.0, L_out=100.0, ds=0.05):
    """Smooth lane change of lateral offset dy over a length L. The curvature is built from three smooth-step
    ramps (0 -> k, k -> -k, -k -> 0 over L/4, L/2, L/4): C1 curvature, zero net heading change.
    k is iterated so that the final lateral offset is exactly dy."""
    def build(k):
        segs = [("const", L_in, 0.0), ("ramp", L/4, 0.0, k), ("ramp", L/2, k, -k), ("ramp", L/4, -k, 0.0),
                ("const", L_out, 0.0)]
        return path_from_segments(segs, ds)
    k = 2*np.pi*dy/L**2
    for _ in range(6):
        k *= dy/build(k)["Y"][-1]
    path = build(k)
    path["type"] = "lanechange"
    return path


def make_path(kind, **kw):
    """Reference paths starting at the origin with heading 0.
      'eight'     : figure eight, keys A_path (450), B_path (300)
      'line'      : straight line of length L
      'uturn'     : straight L_in, 180 deg left arc of radius R, straight L_out
      'sturn'     : straight L_in, left arc (angle alpha, radius R), right arc (same), straight L_out
      'lanechange': lateral offset dy over length L
    Curvature changes follow a smooth step over L_ramp (no curvature jumps)."""
    if kind == "eight":
        return {"type": "eight", "A_path": kw.get("A_path", 450.0), "B_path": kw.get("B_path", 300.0)}
    if kind == "line":
        segs = [("const", kw.get("L", 700.0), 0.0)]
    elif kind == "uturn":
        R, Lr = kw.get("R", 40.0), kw.get("L_ramp", 30.0)
        k0 = 1.0/R
        Lc = (np.pi - k0*Lr)/k0                       # ramps contribute k0*Lr of heading change
        segs = [("const", kw.get("L_in", 80.0), 0.0), ("ramp", Lr, 0.0, k0), ("const", Lc, k0),
                ("ramp", Lr, k0, 0.0), ("const", kw.get("L_out", 150.0), 0.0)]
    elif kind == "sturn":
        R, Lr, alpha = kw.get("R", 50.0), kw.get("L_ramp", 25.0), kw.get("alpha", np.pi/2)
        k0 = 1.0/R
        Lc = alpha/k0 - Lr
        segs = [("const", kw.get("L_in", 60.0), 0.0), ("ramp", Lr, 0.0, k0), ("const", Lc, k0),
                ("ramp", 2*Lr, k0, -k0), ("const", Lc, -k0), ("ramp", Lr, -k0, 0.0),
                ("const", kw.get("L_out", 150.0), 0.0)]
    elif kind == "lanechange":
        return _lane_change_path(**kw)
    else:
        raise ValueError(f"unknown path kind {kind}")
    return path_from_segments(segs, path_type=kind)


def reference_path_geom(theta, path):
    """Reference geometry at path parameter theta: Xr, Yr, psir, kappa, |P'(theta)|.
    Arc-length paths are extrapolated straight beyond their ends."""
    if path.get("type", "eight") == "eight":
        A_path, B_path = path["A_path"], path["B_path"]
        Xr = A_path*np.sin(theta)
        Yr = B_path*np.sin(2.0*theta)
        dX = A_path*np.cos(theta)
        dY = 2.0*B_path*np.cos(2.0*theta)
        ddX = -A_path*np.sin(theta)
        ddY = -4.0*B_path*np.sin(2.0*theta)
        psir = np.arctan2(dY, dX)
        speed = np.hypot(dX, dY)                   # never zero on this path
        kappa = (dX*ddY - dY*ddX)/speed**3
        return Xr, Yr, psir, kappa, speed
    s_g = path["s"]
    th = np.asarray(theta, dtype=float)
    th_c = np.clip(th, s_g[0], s_g[-1])
    over = th - th_c
    psi_c = np.interp(th_c, s_g, path["psi"])
    Xr = np.interp(th_c, s_g, path["X"]) + over*np.cos(psi_c)
    Yr = np.interp(th_c, s_g, path["Y"]) + over*np.sin(psi_c)
    kappa = np.where(over == 0.0, np.interp(th_c, s_g, path["kappa"]), 0.0)
    out = (Xr, Yr, psi_c, kappa, np.ones_like(th))
    return tuple(float(v) for v in out) if th.ndim == 0 else out


def path_length(path, n_grid=20001):
    if path.get("type", "eight") == "eight":
        th = np.linspace(0.0, 2*np.pi, n_grid)
        return np.trapezoid(reference_path_geom(th, path)[4], th)
    return float(path["length"])


def path_quantities(x_HF, theta, path):
    """Quantities at the projection point theta of the truck on the path.
    Frenet progress:  s' = (vx cos psi~ - vy sin psi~)/(1 - kappa e),  theta' = s'/|P'|.
    Disturbance of the LF design model:  d = kappa * vx0.
    Returns Xr, Yr, kappa, psir, d0, theta_dot."""
    e, psi_t, X0, Y0, psi0, psi1, vx0, vy0, r0, r1 = x_HF
    Xr, Yr, psir, kappa, speed = reference_path_geom(theta, path)
    s_dot = (vx0*np.cos(psi_t) - vy0*np.sin(psi_t))/(1.0 - kappa*e)
    theta_dot = s_dot/speed
    d0 = kappa*vx0
    return Xr, Yr, kappa, psir, d0, theta_dot


def initial_vehicle_state(theta0, e_init, vx_init, path):
    """HF state on the path at theta0, shifted laterally by e_init, aligned with the path, speed vx_init."""
    Xr0, Yr0, psir0, _, _ = reference_path_geom(theta0, path)
    X0 = Xr0 - np.sin(psir0)*e_init
    Y0 = Yr0 + np.cos(psir0)*e_init
    psi0 = psi1 = psir0
    e = -np.sin(psir0)*(X0 - Xr0) + np.cos(psir0)*(Y0 - Yr0)
    return np.array([e, psi0 - psir0, X0, Y0, psi0, psi1, vx_init, 0.0, 0.0, 0.0])


def arc_length_interp(theta, path, theta_max):
    """Arc length s [m] and curvature kappa at path parameters `theta`."""
    if path.get("type", "eight") != "eight":                         # theta is already the arc length
        return np.asarray(theta, dtype=float), np.interp(theta, path["s"], path["kappa"])
    th = np.linspace(0.0, theta_max + 1e-6, 40001)
    Xr_, Yr_, psir_, kap_, sp_ = reference_path_geom(th, path)
    s_grid = np.r_[0.0, np.cumsum(0.5*(sp_[1:] + sp_[:-1])*np.diff(th))]
    return np.interp(theta, th, s_grid), np.interp(theta, th, kap_)
