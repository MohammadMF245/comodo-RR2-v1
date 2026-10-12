"""
Performance metrics of closed-loop runs (result dicts from scenarios.run_scenario).
"""
import numpy as np


def rolling_rms(x, n):
    c = np.cumsum(np.r_[0.0, x**2])
    out = np.full(x.size, np.nan)
    out[n-1:] = np.sqrt((c[n:] - c[:-n])/n)
    return out


def tracking_summary(res, settle_time=30.0):
    """Print max steering, max/RMS lateral error after settle_time, integrated vs geometric e."""
    t, e, u, e_geo = res["t"], res["e"], res["u"], res["sig"]["e_geo"]
    settled = t > settle_time
    print("max |steer|           = %.2f deg" % np.rad2deg(np.abs(u).max()))
    if settled.any():
        print("max |e| after %2.0f s    = %.3e m" % (settle_time, np.abs(e[settled]).max()))
        print("RMS e  after %2.0f s     = %.3e m" % (settle_time, np.sqrt(np.mean(e[settled]**2))))
    print("max |e_int - e_geo|   = %.3e m" % np.abs(e - e_geo).max())


def tracking_metrics(res, windows):
    """max |e| and RMS e [cm] in time windows, and RMS of the residual prediction error."""
    t, e, log = res["t"], res["e"], res["log"]
    n = log["u"].size
    out = []
    for (a_, b_) in windows:
        m = (t >= a_) & (t < b_)
        mg = (t[:n] >= a_) & (t[:n] < b_)
        pred_err = log["res_x"][2:5, mg] - log["F_hat_lf"][2:5, mg]
        out.append({"window": (a_, b_),
                    "max_e_cm": 100*np.abs(e[m]).max() if m.any() else np.nan,
                    "rms_e_cm": 100*np.sqrt(np.mean(e[m]**2)) if m.any() else np.nan,
                    "rms_res": np.sqrt(np.mean(pred_err**2, axis=1)) if mg.any() else np.full(3, np.nan)})
    return out


def print_metrics_table(results, windows):
    print(f"{'controller':32s}" + "".join(f"  |e| max / RMS [cm] {a_:>4.0f}-{b_:<4.0f}s" for a_, b_ in windows))
    for r in results:
        m = tracking_metrics(r, windows)
        print(f"{r['label']:32s}" + "".join(f"  {x['max_e_cm']:9.1f} / {x['rms_e_cm']:6.1f}        " for x in m))
    print("\nRMS of the residual left after compensation  (vy0, r0, phi_dot rows of x_LF' [SI]):")
    for r in results:
        m = tracking_metrics(r, [(windows[0][0], windows[-1][1])])[0]
        print(f"  {r['label']:32s} {np.round(m['rms_res'], 4)}")


def performance_metrics(res, vx_ref, t_from=15.0):
    """Tracking / speed / control-effort metrics after the initial transient (t >= t_from)."""
    t, e, log = res["t"], res["e"], res["log"]
    dt = t[1] - t[0]
    m = t >= t_from
    n = log["u"].size
    mg = t[:n] >= t_from
    u = np.rad2deg(log["u"][mg])
    F = log["F_xf"][mg]/1e3                                           # traction force [kN]
    ev = res["Y"][6][m] - vx_ref
    rows = [2, 3, 4]
    err = np.vstack([log["res_x"][rows][:, mg] - log["F_hat_lf"][rows][:, mg],
                     (log["res_v"][mg] - log["F_hat_v"][mg])[None]])
    return {"rms_e": 100*np.sqrt(np.mean(e[m]**2)),
            "rms_ev": np.sqrt(np.mean(ev**2)), "max_ev": np.abs(ev).max(),
            "rms_u": np.sqrt(np.mean(u**2)), "rms_du": np.sqrt(np.mean((np.diff(u)/dt)**2)),
            "rms_F": np.sqrt(np.mean(F**2)), "max_F": np.abs(F).max(), "rms_dF": np.sqrt(np.mean((np.diff(F)/dt)**2)),
            "left": np.sqrt(np.mean(err**2, axis=1))}          # vy0', r0', phi_ddot, vx0' residual left


def print_performance_table(results, vx_ref, t_from=15.0):
    M = [performance_metrics(r, vx_ref, t_from) for r in results]
    b = M[0]
    print(f"(t >= {t_from:.0f} s; percentages relative to '{results[0]['label']}')")
    print(f"{'controller':36s} {'RMS e [cm]':>15s} "
          f"{'speed RMS/max':>14s} {'steer RMS':>9s} {'F_xf RMS/max [kN]':>18s}")
    for r, m in zip(results, M):
        pct = lambda k: f"{m[k]:6.1f} ({100*(m[k]/max(b[k], 1e-12) - 1):+4.0f}%)"
        print(f"{r['label']:36s} {pct('rms_e'):>15s}"
              f"{m['rms_ev']:6.3f}/{m['max_ev']:<6.3f} {m['rms_u']:7.2f}deg {m['rms_F']:8.2f}/{m['max_F']:<8.1f}")
    print("residual left after compensation [SI], rows vy0', r0', phi_ddot, vx0':")
    for r, m in zip(results, M):
        print(f"  {r['label']:36s} {np.round(m['left'], 4)}")


# =============================================================================
# GP covariance monitoring
# =============================================================================
def gp_reliability(res, t_from=0.0):
    """Calibration of the GP covariance: z = (true residual - GP mean)/GP std, coverage, correlation sigma vs |error|."""
    log = res["log"]; n = log["u"].size; m = res["t"][:n] >= t_from
    truth = np.vstack([log["res_x"][2], log["res_x"][3], log["res_x"][4], log["res_v"]])
    mu = log["mu"][[2, 3, 4, 6]]; sd = np.maximum(log["std"][[2, 3, 4, 6]], 1e-12)
    z = (truth - mu)/sd
    out = []
    for j in range(4):
        err = np.abs(truth[j, m] - mu[j, m])
        out.append({"cov1": np.mean(np.abs(z[j, m]) <= 1.0), "cov2": np.mean(np.abs(z[j, m]) <= 2.0),
                    "rms_z": np.sqrt(np.mean(z[j, m]**2)), "rmse": np.sqrt(np.mean(err**2)),
                    "mean_sd": np.mean(sd[j, m]),
                    "corr": np.corrcoef(sd[j, m], err)[0, 1] if sd[j, m].std() > 0 else np.nan})
    return out, z, truth, mu, sd


def print_gp_reliability(results, t_from=0.0):
    print("GP covariance reliability (ideal: coverage 1 sigma = 0.68, 2 sigma = 0.95, RMS z = 1):")
    names = ["vy0'", "r0'", "phi_dd", "vx0'"]
    for r in results:
        if r["mode"] == "none":
            continue
        st = gp_reliability(r, t_from)[0]
        print(f"  {r['label']}")
        for nm, s in zip(names, st):
            print(f"     {nm:7s} cov1 {s['cov1']:.2f}  cov2 {s['cov2']:.2f}  RMS z {s['rms_z']:7.2f}  "
                  f"RMSE {s['rmse']:.4f}  mean sigma {s['mean_sd']:.4f}  corr(sigma,|err|) {s['corr']:5.2f}")


def report(results, vx_ref, t_from=0.0):
    """Performance table + GP reliability."""
    print_performance_table(results, vx_ref, t_from)
    print_gp_reliability(results, t_from)
