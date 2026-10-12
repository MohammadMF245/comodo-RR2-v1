"""
Closed loop: HF plant + backstepping controller + speed controller + GP residual compensation
with online sparse learning.

Closed-loop state layout:
    y = [ x_HF (10) | x_{2..6,d} (5) | xdot_{2..6,d} (5) | xi_{1..5} (5) | theta (1) | w (4) ]

The setting of one run is passed around as plain dicts (no globals):
    ctrl, ref   : controller (controller.design_controller)
    path        : reference path (paths.make_path)
    speed       : {"vx_ref", "vx_dot_ref", "k_vx", "params_ctrl"}
    road        : road.make_road
    gp          : GP options + callables (scenarios.run_scenario builds it)
    mem         : gp_compensation.new_gp_memory
"""
import numpy as np

from .controller import (command_filter_rhs, compensation_rhs, lateral_backstepping_controller,
                         longitudinal_controller, reference_filter_input, reference_filter_output,
                         reference_filter_rhs)
from .gp_compensation import lowpass
from .integrators import rk4_step_aux
from .lf_model import hf_to_lf, lf_to_z
from .paths import path_quantities
from .residual_learning import gp_features_raw, normalize_features, residual_hf_to_lf
from .road import roll_disturbance

N_STATES = 30


# =============================================================================
# State layout
# =============================================================================
def unpack_state(y):
    x_HF = y[0:10]
    xd = np.zeros(7); xdd = np.zeros(7); xi = np.zeros(7)
    xd[2:7] = y[10:15]; xdd[2:7] = y[15:20]; xi[1:6] = y[20:25]
    return x_HF, xd, xdd, xi, y[25], y[26:30]


def pack_state(x_HF, xd, xdd, xi, theta, w):
    y = np.zeros(N_STATES)
    y[0:10] = x_HF; y[10:15] = xd[2:7]; y[15:20] = xdd[2:7]; y[20:25] = xi[1:6]
    y[25] = theta; y[26:30] = w
    return y


# =============================================================================
# GP evaluation (zero-order hold over one step)
# =============================================================================
def update_F_hat(t, y, mem, ctrl, path, speed, road, gp, F_hat_fn, dt=0.0, refresh=True):
    """Evaluate the GP with the PREVIOUS control input, map x -> z (F_hat_z = T F_hat_x) and store it
    in mem (zero-order hold over the step).

    refresh=False re-uses the last raw GP output (GP evaluated every gp['gp_every'] steps) but the
    low-pass filter on F_hat is still advanced.

    Low-pass filtering F_hat (tau_F ~ 1-2 s) turns it into a slowly varying disturbance estimate and
    keeps the loop u -> F_hat -> u (one step delay) stable."""
    if refresh or mem.get("raw") is None:
        x_HF, _, _, _, theta, _ = unpack_state(y)
        _, _, kappa, _, _, _ = path_quantities(x_HF, theta, path)
        F_xf = mem["F_xf_prev"] if mem.get("F_xf_prev") is not None else \
            longitudinal_controller(x_HF, speed["vx_ref"], speed["vx_dot_ref"],
                                    speed["k_vx"], speed["params_ctrl"])     # previous APPLIED force
        phi_roll = float(road["phi_roll_fn"](t, x_HF, theta))
        d_roll_ctrl = roll_disturbance(phi_roll, road["g"]) if road["feedforward"] else 0.0
        mem["u_gp"] = lowpass(mem["u_gp"], mem["u_prev"], dt, gp.get("tau_u", 0.0))
        gp_in = {"delta": mem["u_gp"], "F_xf": F_xf, "d_roll": d_roll_ctrl, "kappa": kappa}
        _, mem["raw"] = F_hat_fn(t, x_HF, gp_in, ctrl["T"], gp)
    info = mem["raw"]
    F_lf = info["F_hat_lf"] if mem["F_hat_lf"] is None else \
        lowpass(mem["F_hat_lf"], info["F_hat_lf"], dt, gp.get("tau_F", 0.0))
    mem["F_hat_lf"] = F_lf
    mem["F_hat"] = ctrl["T"] @ F_lf                      # x_LF -> z domain
    Fv = float(info.get("F_hat_v", 0.0))                 # longitudinal residual, same low-pass
    mem["F_hat_v"] = Fv if mem["F_hat_v"] is None else lowpass(mem["F_hat_v"], Fv, dt, gp.get("tau_F", 0.0))
    mem["info"] = dict(info, F_hat_lf=F_lf)
    return mem["F_hat"], mem["info"]


# =============================================================================
# Signals and right-hand side
# =============================================================================
def closed_loop_signals(t, y, ctrl, ref, path, speed, road, mem):
    """All controller signals at (t, y). F_hat_z is taken from mem (held over the step)."""
    x_HF, xd, xdd, xi, theta, w = unpack_state(y)
    Xr, Yr, kappa, psir, d0, theta_dot = path_quantities(x_HF, theta, path)

    z = lf_to_z(hf_to_lf(x_HF), ctrl["T"])

    phi_roll = float(road["phi_roll_fn"](t, x_HF, theta))                     # acts on the plant
    d_roll_plant = roll_disturbance(phi_roll, road["g"])
    d_roll = d_roll_plant if road["feedforward"] else 0.0                       # used by controller

    F_hat = np.asarray(mem["F_hat"], dtype=float)                               # z-coordinates

    g = reference_filter_input(d0, d_roll, F_hat, ctrl["c"], ctrl["c_roll"], ref["use_F_hat"])
    x1d, x1d_dot = reference_filter_output(w, g, ref)

    alpha, e, v = lateral_backstepping_controller(
        z, xd, xdd, xi, d0, d_roll, x1d, x1d_dot, F_hat,
        ctrl["C"], ctrl["a"], ctrl["b"], ctrl["c"], ctrl["c_roll"])
    F_hat_v = 0.0 if mem.get("F_hat_v") is None else float(mem["F_hat_v"])   # learned vx0' residual
    F_xf = longitudinal_controller(x_HF, speed["vx_ref"], speed["vx_dot_ref"],
                                   speed["k_vx"], speed["params_ctrl"], res_vx_hat=F_hat_v)
    return dict(x_HF=x_HF, xd=xd, xdd=xdd, xi=xi, theta=theta, w=w, g=g,
                Xr=Xr, Yr=Yr, kappa=kappa, psir=psir, d0=d0, theta_dot=theta_dot,
                phi_roll=phi_roll, d_roll=d_roll, d_roll_plant=d_roll_plant,
                z=z, F_hat=F_hat, F_hat_v=F_hat_v, x1d=x1d, alpha=alpha, e=e, v=v, u=alpha[6], F_xf=F_xf)


def closed_loop_rhs(t, y, model_HF, ctrl, ref, path, speed, road, mem):
    s = closed_loop_signals(t, y, ctrl, ref, path, speed, road, mem)
    dx_HF, tf_HF = model_HF.full_dynamics(s["x_HF"], s["u"], s["F_xf"], s["phi_roll"],
                                          kappa=s["kappa"], psir=s["psir"])
    dxd, dxdd = command_filter_rhs(s["xd"], s["xdd"], s["alpha"], ctrl["wf"], ctrl["zeta"])
    dy = np.zeros_like(y)
    dy[0:10] = dx_HF
    dy[10:15] = dxd
    dy[15:20] = dxdd
    dy[20:25] = compensation_rhs(s["xd"], s["xi"], s["alpha"], ctrl["C"])
    dy[25] = s["theta_dot"]
    dy[26:30] = reference_filter_rhs(s["w"], s["g"], ref)
    return {'dy': dy, 'dx_HF': dx_HF, 'tf': tf_HF, 'CL_signals': s}


# =============================================================================
# Simulation with GP update and previous-input memory
# =============================================================================
def simulate_closed_loop(y0, t_grid, model_HF, ctrl, ref, path, speed, road, gp, F_hat_fn, mem):
    """Fixed-step RK4 of the closed loop with GP residual compensation and online GP learning.

    Every step k:
      1. F_hat_z(k) = T @ GP(x_k, u_{k-1}, F_xf, d_r, kappa)   (GP evaluated every gp['gp_every'] steps,
         low-pass filtered with tau_F, held over the RK4 stages)
      2. RK4 step; u_k = control input at (t_k, y_k)
      3. true residual (HF - LF) at (x_k, u_k); every gp['learn_every'] steps it is passed to
         gp['learn_fn'](x_feature_normalised, residual)
      4. mem['u_prev'] = u_k
    Returns t, Y and a log dict (per step k = 0..N-1)."""
    n_steps = len(t_grid) - 1
    Y = np.zeros((len(y0), len(t_grid)))
    Y[:, 0] = y0
    log = {"u": np.zeros(n_steps), "u_prev": np.zeros(n_steps), "u_gp": np.zeros(n_steps),
           "F_hat_z": np.zeros((6, n_steps)), "F_hat_lf": np.zeros((6, n_steps)),
           "mu": np.zeros((7, n_steps)), "std": np.zeros((7, n_steps)),
           "res_x": np.zeros((6, n_steps)), "res_v": np.zeros(n_steps),
           "x_feature_true": np.zeros((11, n_steps)), "n_learned": np.zeros(n_steps),
           "F_hat_v": np.zeros(n_steps), "F_xf": np.zeros(n_steps),
           "trace_Sig": np.zeros((6, n_steps)), "ood": np.zeros(n_steps),
           "pool": np.zeros(n_steps)}
    args = (model_HF, ctrl, ref, path, speed, road, mem)
    gp_every, learn_every = gp.get("gp_every", 1), gp.get("learn_every", 1)
    n_learned = 0
    for k in range(n_steps):
        dt = t_grid[k+1] - t_grid[k]

        # 1. GP prediction with the previous control input
        F_hat_z, info = update_F_hat(t_grid[k], Y[:, k], mem, ctrl, path, speed, road, gp, F_hat_fn, dt,
                                     refresh=(k % gp_every == 0))

        # 2. integrate one step (F_hat held constant in all RK4 stages)
        CL_rhs, Y[:, k+1] = rk4_step_aux(closed_loop_rhs, t_grid[k], Y[:, k], dt, args)
        s = CL_rhs['CL_signals']
        u_k = s["u"]

        # 3. true residual at the applied inputs.
        #    gp['residual_bank'] = 'plant'     : bank acting on the truck is part of the LF model
        #                          'controller': only the bank the CONTROLLER knows is in the LF model, so an
        #                                        unknown bank shows up in the residual and can be learned
        d_roll_res = s["d_roll"] if gp.get("residual_bank", "plant") == "controller" else s["d_roll_plant"]
        res_x, res_v = residual_hf_to_lf(Y[:10, k], u_k, s["F_xf"], d_roll_res, s["d0"],
                                         speed["params_ctrl"], CL_rhs['dx_HF'], ctrl['LF'])
        feat_raw = gp_features_raw(Y[:10, k], u_k, s["F_xf"], d_roll_res, s["kappa"])

        # online sparse GP update
        if gp.get("learn_fn") is not None and k % learn_every == 0:
            gp["learn_fn"](normalize_features(feat_raw, gp["normalizer"]), np.r_[res_x, res_v])
            n_learned += 1

        log["u"][k] = u_k; log["u_prev"][k] = mem["u_prev"]; log["u_gp"][k] = mem["u_gp"]
        log["F_hat_z"][:, k] = F_hat_z
        log["F_hat_lf"][:, k] = info["F_hat_lf"]
        log["mu"][:, k] = info["mu"]; log["std"][:, k] = info["std"]
        log["res_x"][:, k] = res_x; log["res_v"][k] = res_v
        log["x_feature_true"][:, k] = feat_raw
        log["n_learned"][k] = n_learned
        log["F_hat_v"][k] = s["F_hat_v"]; log["F_xf"][k] = s["F_xf"]
        log["ood"][k] = float(info.get("ood", False))
        if gp.get("monitor_fn") is not None:
            log["trace_Sig"][:, k] = gp["monitor_fn"]()      # trace of the parametric posterior covariance
        if gp.get("pool_size_fn") is not None:
            log["pool"][k] = gp["pool_size_fn"]()

        # 4. remember the applied inputs for the next GP evaluation
        mem["u_prev"] = float(u_k)
        mem["F_xf_prev"] = float(s["F_xf"])

        if not np.all(np.isfinite(Y[:, k+1])):
            print(f"WARNING: simulation diverged at t = {t_grid[k+1]:.2f} s")
            return t_grid[:k+1], Y[:, :k+1], {key: val[..., :k] for key, val in log.items()}
    return t_grid, Y, log


# =============================================================================
# Post-processing
# =============================================================================
def post_process(t, Y, ctrl, ref, path, speed, road, log):
    """Signals along the trajectory, re-evaluated with the F_hat_z actually used in each step."""
    keys = ["u", "F_xf", "d0", "kappa", "x1d", "Xr", "Yr", "phi_roll", "d_roll", "psir"]
    out = {k: np.zeros(len(t)) for k in keys}
    out["F_hat"] = np.zeros((6, len(t)))
    out["e_geo"] = np.zeros(len(t))
    n_log = log["F_hat_z"].shape[1]
    for k in range(len(t)):
        mem_k = {"F_hat": log["F_hat_z"][:, min(k, n_log - 1)]}
        s = closed_loop_signals(t[k], Y[:, k], ctrl, ref, path, speed, road, mem_k)
        for key in keys:
            out[key][k] = s[key]
        out["F_hat"][:, k] = s["F_hat"]
        x = s["x_HF"]                                  # geometric check of integrated e
        out["e_geo"][k] = -np.sin(s["psir"])*(x[2] - s["Xr"]) + np.cos(s["psir"])*(x[3] - s["Yr"])
    return out
