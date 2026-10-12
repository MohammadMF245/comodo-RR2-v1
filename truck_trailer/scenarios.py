"""
Scenario runner: the same plant with / without GP, with / without online learning.

    run_scenario(mode, ...)   one closed-loop run; mode = 'none' | 'gp_fixed' | 'gp_online'
    run_cases(cases, ...)     a list of controller cases on one plant / path
    standard_cases(...)       no GP / offline GP / online GP / online GP + vx0' residual
"""
import copy
import time

import numpy as np

from .closed_loop import pack_state, post_process, simulate_closed_loop, update_F_hat
from .controller import initial_controller_states
from .gp_compensation import F_hat_gp, F_hat_zero, new_gp_memory
from .hf_model import TruckTrailerDynamics
from .paths import initial_vehicle_state, path_length
from .road import make_road
from .spgp import make_f_pred_spgp, spgp_bank_trace, spgp_bank_update

DEFAULT_GP_OPTS = {"tau_F": 2.0,        # [s] low-pass on F_hat (slowly varying disturbance estimate)
                   "tau_u": 0.0,        # [s] low-pass on the previous input fed to the GP
                   "gp_every": 4,       # evaluate the GP every 4 steps (zero-order hold)
                   "learn_every": 20,   # online sparse update every 20 steps (0.1 s at dt_s = 0.005)
                   "rows_used": np.array([1, 1, 1, 1, 1, 0]),  # x_LF rows e, psi~, vy0, r0, phi_dot, phi
                   "max_std": None,     # e.g. 0.05: no compensation where the GP is uncertain
                   "gain": 1.0,
                   "gain_vx": 0.5,      # gain on the learned vx0' residual
                   "ood_limit": None,   # e.g. 4.0: GP off when an input is > 4 std outside the training data
                   "forget": 1.0,       # < 1: forgetting factor on the parametric covariance (faster adaptation)
                   "use_vx_residual": False,   # True: learned vx0' residual corrects the speed controller
                   # residual definition: 'controller' = HF minus the LF model WITH ONLY the disturbances the
                   # controller knows (an unknown road bank then appears in the residual and can be learned);
                   # 'plant' = HF minus LF with the true bank (bank never learned)
                   "residual_bank": "controller"}


def make_speed(vx_ref, params_ctrl, k_vx=1.0, vx_dot_ref=0.0):
    """Speed-loop settings (the controller only knows the nominal parameters params_ctrl)."""
    return {"vx_ref": vx_ref, "vx_dot_ref": vx_dot_ref, "k_vx": k_vx, "params_ctrl": params_ctrl}


def _build_gp(mode, bank, normalizer, opts):
    if mode == "none":
        return F_hat_zero, {"tau_F": 0.0, "tau_u": 0.0, "gp_every": 1, "learn_every": 1, "learn_fn": None}
    gp = {"f_pred": make_f_pred_spgp(bank), "normalizer": normalizer,
          "rows_used": opts["rows_used"], "max_std": opts["max_std"], "gain": opts["gain"],
          "tau_u": opts["tau_u"], "tau_F": opts["tau_F"], "gp_every": opts["gp_every"],
          "learn_every": opts["learn_every"], "use_vx_residual": opts["use_vx_residual"],
          "ood_limit": opts["ood_limit"], "ood_mask": bank["mask"], "gain_vx": opts["gain_vx"],
          "residual_bank": opts["residual_bank"],
          "learn_fn": (lambda x, y: spgp_bank_update(bank, x, y, forget=opts["forget"])) if mode == "gp_online" else None,
          "pool_size_fn": lambda: bank["states"][0]["alpha"].size,
          "monitor_fn": lambda: spgp_bank_trace(bank)}
    return F_hat_gp, gp


def run_scenario(mode, sp_bank0, normalizer, ctrl, ref, path, speed, road, params_HF, tire_model,
                 dt_s, laps, e_init, gp_opts=None, label=None, verbose=True, T_end=None, vx_init=None,
                 theta0=0.0):
    """Closed-loop run. mode: 'none' (F_hat = 0) | 'gp_fixed' (offline-trained GP) |
    'gp_online' (GP + online sparse updates). sp_bank0 is deep-copied, so runs are independent.

    Returns dict(mode, label, t, Y, log, sig, bank, e, u)."""
    opts = dict(DEFAULT_GP_OPTS, **(gp_opts or {}))
    bank = copy.deepcopy(sp_bank0) if mode != "none" else None
    F_hat_fn, gp = _build_gp(mode, bank, normalizer, opts)

    model_HF = TruckTrailerDynamics(params_HF, tire_model=tire_model)
    mem = new_gp_memory(u_prev0=0.0)

    x0_HF = initial_vehicle_state(theta0, e_init, speed["vx_ref"] if vx_init is None else vx_init, path)
    y0 = pack_state(x0_HF, np.zeros(7), np.zeros(7), np.zeros(7), theta0, np.zeros(4))
    update_F_hat(0.0, y0, mem, ctrl, path, speed, road, gp, F_hat_fn)       # F_hat(0) with u_prev0
    xd0, xdd0, xi0, w0 = initial_controller_states(0.0, x0_HF, theta0, ctrl, ref, path, road, mem["F_hat"])
    y0 = pack_state(x0_HF, xd0, xdd0, xi0, theta0, w0)

    if T_end is None:
        T_end = laps*path_length(path)/speed["vx_ref"] + 5.0
    t_grid = np.arange(int(np.ceil(T_end/dt_s)) + 1)*dt_s
    t0 = time.time()
    t, Y, log = simulate_closed_loop(y0, t_grid, model_HF, ctrl, ref, path, speed, road, gp, F_hat_fn, mem)
    sig = post_process(t, Y, ctrl, ref, path, speed, road, log)
    if verbose:
        print(f"[{label or mode}] {len(t)} steps in {time.time() - t0:.0f} s")
    return {"mode": mode, "label": label or mode, "t": t, "Y": Y, "log": log, "sig": sig, "bank": bank,
            "e": Y[0], "u": sig["u"]}


def run_cases(cases, banks, normalizer, ctrl, ref, path, speed, params_HF, tire_model, dt_s, e_init,
              gp_opts=None, laps=1.0, T_end=None, vx_init=None):
    """Run a list of controller cases on one plant / path. Each case is a dict:
       mode ('none' | 'gp_fixed' | 'gp_online'), label, bank (key in `banks` or None),
       road (road.make_road), opts (GP options overriding gp_opts)."""
    out = []
    for cs in cases:
        res = run_scenario(cs["mode"], (banks or {}).get(cs.get("bank")), normalizer, ctrl, ref, path, speed,
                           cs["road"], params_HF, tire_model, dt_s, laps, e_init,
                           gp_opts={**(gp_opts or {}), **cs.get("opts", {})},
                           label=cs["label"], T_end=T_end, vx_init=vx_init)
        res["controller_aware"] = bool(cs["road"]["feedforward"])      # the controller knows the road bank
        res["speed_loop_on"] = speed.get("k_vx", 1.0) > 0.0            # the longitudinal controller is active
        out.append(res)
    return out


# =============================================================================
# Case lists used in the experiments
# =============================================================================
def standard_cases(road, vx_gains=(0.5,), fixed=True, bank="aware"):
    """no GP / GP trained offline only / GP + online sparse updates / + learned vx0' residual in the speed loop."""
    cases = [dict(mode="none", label="no GP", road=road)]
    if fixed:
        cases.append(dict(mode="gp_fixed", label="GP (offline only)", bank=bank, road=road))
    cases.append(dict(mode="gp_online", label="GP online (lateral)", bank=bank, road=road))
    for g in vx_gains:
        cases.append(dict(mode="gp_online", label=f"GP online + vx0' residual (gain {g:g})", bank=bank, road=road,
                          opts={"use_vx_residual": True, "gain_vx": g}))
    return cases


def bank_cases(bank_rad, forget=0.95):
    """Unknown road bank: no GP (controller unaware) / no GP (controller aware, feedforward) / online GP unaware."""
    return [dict(mode="none", label="no GP, controller unaware", road=make_road(bank_rad, False)),
            dict(mode="none", label="no GP, controller aware (feedforward)", road=make_road(bank_rad, True)),
            dict(mode="gp_online", label="GP unaware", bank="unaware", road=make_road(bank_rad, False),
                 opts={"forget": forget})]


def unaware_extra_cases(bank_rad, forget=0.95):
    """Unaware controller, speed loop on: online GP + vx0' residual (used by the bar chart)."""
    return [dict(mode="gp_online", label="GP online + vx0' residual", bank="unaware", road=make_road(bank_rad, False),
                 opts={"forget": forget, "use_vx_residual": True})]
