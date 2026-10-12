"""
Quick-look plots of a single run or a comparison (interactive use).
Publication-quality SVG figures are in `figures.py`.
"""
import matplotlib.pyplot as plt
import numpy as np

from .metrics import rolling_rms, tracking_metrics
from .paths import reference_path_geom


def plot_tracking_overview(res, path, n_ref=2000):
    """2x2: x-y path vs reference | lateral error | steering | angles and disturbances."""
    t, x, sig = res["t"], res["Y"][0:10, :], res["sig"]
    if path.get("type", "eight") == "eight":
        th = np.linspace(0, 2*np.pi, n_ref)
        Xp, Yp = reference_path_geom(th, path)[:2]
    else:
        Xp, Yp = path["X"], path["Y"]
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(Xp, Yp, "k--", lw=1, label="reference")
    ax.plot(x[2], x[3], lw=1.5, label="truck (HF)")
    ax.plot(x[2, 0], x[3, 0], "o", label="start")
    ax.set_aspect("equal"); ax.grid(True); ax.legend(loc="lower center")
    ax.set_title("Path tracking"); ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(t, x[0], label="e (integrated)"); ax.plot(t, sig["e_geo"], "--", label="e (geometric)")
    ax.grid(True); ax.legend(loc="upper right"); ax.set_title("Lateral error")
    ax.set_xlabel("t [s]"); ax.set_ylabel("e [m]")

    ax = fig.add_subplot(2, 2, 3)
    ax.plot(t, np.rad2deg(res["u"])); ax.grid(True)
    ax.set_title(r"Steering $\delta$"); ax.set_xlabel("t [s]"); ax.set_ylabel("deg")

    ax = fig.add_subplot(2, 2, 4)
    ax.plot(t, np.rad2deg(x[5] - x[4]), label=r"articulation $\phi$")
    ax.plot(t, np.rad2deg(x[1]), label=r"heading error $\tilde\psi$")
    ax.plot(t, np.rad2deg(sig["phi_roll"]), label=r"road bank $\phi_{roll}$")
    ax.set_ylabel("deg"); ax.grid(True); ax.set_xlabel("t [s]")
    ax2 = ax.twinx(); ax2.plot(t, sig["d0"], "k:", lw=1, label=r"$d=\kappa v_{x0}$"); ax2.set_ylabel("d [rad/s]")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right"); ax.set_title("Angles and disturbances")
    plt.tight_layout()
    return fig


def plot_hf_states(res):
    """Control u, Cartesian error and a 5x2 grid of the HF states of one run."""
    t = res["t"]
    states = res["Y"][2:10].T                         # x0, y0, psi0, psi1, vx0, vy0, r0, r1
    ref = np.vstack([res["sig"]["Xr"], res["sig"]["Yr"]]).T
    figs = []

    fig = plt.figure(); figs.append(fig)
    plt.plot(t, np.rad2deg(res["u"])); plt.title("Control u")
    plt.xlabel("Time [s]"); plt.ylabel(r"$u$ [deg]"); plt.grid(True)

    fig = plt.figure(); figs.append(fig)
    plt.plot(t, states[:, 0] - ref[:, 0], label="e_x")
    plt.plot(t, states[:, 1] - ref[:, 1], label="e_y")
    plt.title("Error"); plt.xlabel("Time [s]"); plt.ylabel("$e$ [m]"); plt.grid(True); plt.legend()

    fig = plt.figure(figsize=(14, 18)); figs.append(fig)
    ax0 = plt.subplot(5, 2, 1)
    ax0.plot(states[:, 0], states[:, 1], linewidth=2, label="Truck trajectory-HF")
    ax0.plot(ref[:, 0], ref[:, 1], '-.', linewidth=1, label="Ref.")
    ax0.plot(states[0, 0], states[0, 1], 'o', label="Start")
    ax0.plot(states[-1, 0], states[-1, 1], 's', label="End-HF")
    ax0.set_title("Truck Trajectory"); ax0.set_xlabel("$x_0$ [m]"); ax0.set_ylabel("$y_0$ [m]")
    ax0.axis("equal"); ax0.grid(True); ax0.legend()

    panels = [  # (subplot, column, title, ylabel, scale, with reference column)
        (3, 0, "Truck Longitudinal Position $x_0$", "$x_0$ [m]", 1.0, 0),
        (4, 1, "Truck Lateral Position $y_0$", "$y_0$ [m]", 1.0, 1),
        (5, 2, "Truck Yaw Angle $\\psi_0$", "$\\psi_0$ [deg]", None, None),
        (6, 3, "Trailer Yaw Angle $\\psi_1$", "$\\psi_1$ [deg]", None, None),
        (7, 4, "Longitudinal Velocity $v_{x0}$", "$v_{x0}$ [m/s]", 1.0, None),
        (8, 5, "Lateral Velocity $v_{y0}$", "$v_{y0}$ [m/s]", 1.0, None),
        (9, 6, "Truck Yaw Rate $r_0$", "$r_0$ [deg/s]", None, None),
        (10, 7, "Trailer Yaw Rate $r_1$", "$r_1$ [deg/s]", None, None),
    ]
    for sp, col, title, ylabel, scale, refcol in panels:
        ax = plt.subplot(5, 2, sp)
        y = states[:, col] if scale is not None else np.rad2deg(states[:, col])
        ax.plot(t, y, label="HF")
        if refcol is not None:
            ax.plot(t, ref[:, refcol], '-.', label="Ref.")
        ax.set_title(title); ax.set_xlabel("Time [s]"); ax.set_ylabel(ylabel); ax.grid(True); ax.legend()
    plt.tight_layout()
    return figs


def plot_gp_comparison(results, windows, win_s=20.0):
    """Lateral error, rolling RMS, residual left, steering, per-window bars and online-GP diagnostics."""
    colors = {"none": "tab:red", "gp_fixed": "tab:orange", "gp_online": "tab:green"}
    fig, axs = plt.subplots(4, 2, figsize=(14, 15))
    for r in results:
        t, log, col = r["t"], r["log"], colors[r["mode"]]
        n = log["u"].size
        dt = t[1] - t[0]; nw = int(win_s/dt)
        axs[0, 0].plot(t, 100*r["e"], color=col, lw=0.9, label=r["label"])
        axs[0, 1].semilogy(t, 100*rolling_rms(r["e"], nw), color=col, label=r["label"])
        for j, (row, name) in enumerate([(2, r"$\dot v_{y0}$"), (3, r"$\dot r_0$")]):
            err = log["res_x"][row] - log["F_hat_lf"][row]
            axs[1, j].semilogy(t[:n], rolling_rms(err, nw), color=col, label=r["label"])
            axs[1, j].set_title(f"RMS of {name} residual left after compensation ({win_s:.0f} s window)")
        axs[2, 0].plot(t[:n], np.rad2deg(log["u"]), color=col, lw=0.8, label=r["label"])
    axs[0, 0].set_title("Lateral error e"); axs[0, 0].set_ylabel("e [cm]")
    axs[0, 1].set_title(f"Rolling RMS of e ({win_s:.0f} s window)"); axs[0, 1].set_ylabel("[cm]")
    axs[2, 0].set_title("Steering angle"); axs[2, 0].set_ylabel("deg")
    for ax in axs.ravel()[:5]:
        ax.grid(True, which="both"); ax.legend(fontsize=8); ax.set_xlabel("t [s]")
    ax = axs[2, 1]; width = 0.8/len(results)
    for j, r in enumerate(results):
        m = tracking_metrics(r, windows)
        x = np.arange(len(windows)) + j*width
        ax.bar(x, [q["rms_e_cm"] for q in m], width, color=colors[r["mode"]], label=r["label"] + " (RMS)")
        ax.plot(x, [q["max_e_cm"] for q in m], "k_", markersize=12)
    ax.set_xticks(np.arange(len(windows)) + 0.4 - width/2)
    ax.set_xticklabels([f"{a_:.0f}-{b_:.0f} s" for a_, b_ in windows])
    ax.set_ylabel("|e| [cm]  (bars: RMS, ticks: max)"); ax.set_title("Lateral error per time window")
    ax.grid(True, axis="y"); ax.legend(fontsize=8)
    ro = [r for r in results if r["mode"] == "gp_online"]
    if ro:
        r = ro[0]; log = r["log"]; n = log["u"].size
        axs[3, 0].plot(r["t"][:n], log["res_x"][2], lw=0.8, label=r"true residual $\dot v_{y0}$")
        axs[3, 0].plot(r["t"][:n], log["F_hat_lf"][2], lw=0.8, label=r"$\hat F$ used (filtered, online GP)")
        axs[3, 0].set_title("Vy0 residual: truth vs compensation (online GP)"); axs[3, 0].legend(fontsize=8)
        axs[3, 0].grid(True); axs[3, 0].set_xlabel("t [s]")
        ax = axs[3, 1]; ax.plot(r["t"][:n], log["n_learned"], color="tab:blue")
        ax.set_ylabel("online updates absorbed", color="tab:blue"); ax.set_xlabel("t [s]")
        ax2 = ax.twinx(); ax2.plot(r["t"][:n], log["pool"], color="tab:purple")
        ax2.set_ylabel("pool size (sparse)", color="tab:purple")
        ax.set_title("Online sparse GP: data absorbed vs pool size"); ax.grid(True)
    plt.tight_layout()
    return fig
