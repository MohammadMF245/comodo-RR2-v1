"""
Publication figures: SVG, one file per plot, LaTeX-style labels.

    save_error_figures(scenario_runs, vx_ref, out_dir)   tracking-error plots per scenario + RMS bar chart
    save_state_figures(scenario_runs, out_dir)           HF states: "no GP" vs the best online GP

`scenario_runs` maps "<n>. <description>" -> dict(results=[...] or results_by_v0={v0: [...]},
extra_results=[...], path=..., t_from=..., win_s=...). Scenario keys "1".."4" are
infinity / U-turn / S-shape / straight line. Call `configure(use_tex=True)` to typeset with LaTeX.
"""
import os
import re

import matplotlib.pyplot as plt
import numpy as np

CONFIG = {"use_tex": False,            # True: system LaTeX (needs latex); False: matplotlib mathtext
          "png_preview": False,        # also write a .png next to every .svg
          "t_from": 0.0}               # plots / RMS use t >= t_from


def configure(**kw):
    CONFIG.update(kw)


def pub_rc():
    rc = {"text.usetex": CONFIG["use_tex"], "font.family": "serif", "mathtext.fontset": "cm", "svg.fonttype": "path",
          "axes.labelsize": 11, "axes.titlesize": 11, "legend.fontsize": 8.5, "xtick.labelsize": 9,
          "ytick.labelsize": 9, "axes.grid": True, "grid.alpha": 0.3, "lines.linewidth": 1.3}
    if CONFIG["use_tex"]:
        rc["text.latex.preamble"] = r"\usepackage{amsmath}"
    return rc


T_LABEL = r"$t\ [\mathrm{s}]$"

TEX_LABEL = {
    "no GP": r"no GP",
    "GP (offline only)": r"GP (offline only)",
    "GP online (lateral)": r"GP online",
    "GP online + vx0' residual (gain 0.5)": r"GP online $+\ \dot{v}_{x0}$ residual",
    "GP online + vx0' residual": r"GP online $+\ \dot{v}_{x0}$ residual",
    "no GP, controller unaware": r"no GP, controller unaware of $\phi_{\text{roll}}$",
    "no GP, controller aware (feedforward)": r"no GP, controller aware of $\phi_{\text{roll}}$",
    "no GP, controller aware": r"no GP, controller aware of $\phi_{\text{roll}}$",
    "GP unaware": r"GP unaware of $\phi_{\text{roll}}$",
    "GP aware + controller aware": r"GP aware, controller aware of $\phi_{\text{roll}}$",
    "no GP, controller aware, speed loop OFF": r"no GP, controller aware of $\phi_{\text{roll}}$, speed loop off",
}
ROLE_STYLE = {"baseline": ("tab:red", "-"), "nogp_aware": ("black", "--"), "offline": ("tab:orange", "-"),
              "online": ("tab:green", "-"), "online_vx": ("tab:blue", "-"), "gp_unaware": ("tab:cyan", "-"),
              "gp_aware": ("tab:purple", "-"), "speed_off": ("tab:brown", ":")}
ROLE_TEXT = {"baseline": r"no GP", "offline": r"GP (offline only)", "online": r"GP online",
             "online_vx": r"GP online $+\ \dot{v}_{x0}$ residual"}
BAR_ORDER = ["baseline", "offline", "online", "online_vx"]


# =============================================================================
# Helpers
# =============================================================================
def _esc(s):
    return re.sub(r"([%&_#])", r"\\\1", s) if CONFIG["use_tex"] else s


def tex_label(label):
    return TEX_LABEL.get(label, _esc(label))


def case_role(r):
    """Role of a run from its label (sets colour / legend)."""
    lab = r["label"]
    if "speed loop OFF" in lab:
        return "speed_off"
    if lab.startswith("no GP"):
        return "nogp_aware" if ("aware" in lab and "unaware" not in lab) else "baseline"
    if "vx0" in lab:
        return "online_vx"
    if "offline" in lab:
        return "offline"
    if "GP aware" in lab:
        return "gp_aware"
    if "GP unaware" in lab:
        return "gp_unaware"
    return "online"


def bar_role(role):
    """The unaware GP that learns online counts as 'GP online'."""
    return "online" if role == "gp_unaware" else role


def _style(r):
    return ROLE_STYLE[case_role(r)]


def _controller_aware(r):
    if "controller_aware" in r:
        return r["controller_aware"]
    return "aware" in r["label"] and "unaware" not in r["label"]


def _speed_loop_on(r):
    return r.get("speed_loop_on", "OFF" not in r["label"])


def _stride(n, max_pts=3000):
    return max(1, int(np.ceil(n/max_pts)))


def _rolling_rms(x, n):
    c = np.cumsum(np.r_[0.0, x**2])
    out = np.full(x.size, np.nan)
    out[n-1:] = np.sqrt((c[n:] - c[:-n])/n)
    return out


def _xy_err(r):
    """Cartesian error components X0 - Xr, Y0 - Yr [cm]."""
    return 100*(r["Y"][2] - r["sig"]["Xr"]), 100*(r["Y"][3] - r["sig"]["Yr"])


def _rms_e(r):
    m = r["t"] >= CONFIG["t_from"]
    return np.sqrt(np.mean(r["e"][m]**2))


def scenario_results(sc):
    return sc["results"] if "results" in sc else list(sc["results_by_v0"].values())[0]


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name + ".svg")
    fig.savefig(path, format="svg", bbox_inches="tight")
    if CONFIG["png_preview"]:
        fig.savefig(os.path.join(out_dir, name + ".png"), dpi=110, bbox_inches="tight")
    print("saved", path)
    return path


def _new(figsize=(6.4, 3.6), nrows=1, ncols=1, **kw):
    return plt.subplots(nrows, ncols, figsize=figsize, **kw)


# =============================================================================
# Error figures
# =============================================================================
def plot_cart_norm(res, prefix, out_dir):
    fig, ax = _new()
    for r in res:
        t = r["t"]; ex, ey = _xy_err(r); m = t >= CONFIG["t_from"]; s = _stride(m.sum()); col, ls = _style(r)
        ax.plot(t[m][::s], np.hypot(ex, ey)[m][::s], color=col, ls=ls, label=tex_label(r["label"]))
    ax.set_xlabel(T_LABEL); ax.set_ylabel(r"$\Vert (X_0-X_r,\ Y_0-Y_r)\Vert\ [\mathrm{cm}]$"); ax.legend()
    return _save(fig, out_dir, f"{prefix}_cartesian_error_norm")


def plot_abs_e(res, prefix, out_dir):
    fig, ax = _new()
    for r in res:
        t = r["t"]; m = t >= CONFIG["t_from"]; s = _stride(m.sum()); col, ls = _style(r)
        ax.plot(t[m][::s], 100*np.abs(r["e"])[m][::s], color=col, ls=ls, label=tex_label(r["label"]))
    ax.set_xlabel(T_LABEL); ax.set_ylabel(r"$|e|\ [\mathrm{cm}]$"); ax.legend()
    return _save(fig, out_dir, f"{prefix}_lateral_error_norm")


def plot_rolling_rms(res, prefix, win_s, out_dir):
    fig, ax = _new()
    dt = res[0]["t"][1] - res[0]["t"][0]; n = max(int(round(win_s/dt)), 1)
    for r in res:
        t = r["t"]; y = _rolling_rms(100*r["e"], n); m = np.isfinite(y) & (t >= CONFIG["t_from"])
        s = _stride(m.sum()); col, ls = _style(r)
        ax.plot(t[m][::s], y[m][::s], color=col, ls=ls, label=tex_label(r["label"]))
    ax.set_xlabel(T_LABEL)
    ax.set_ylabel(r"rolling RMS of $e$ (window $%g\,\mathrm{s}$) $[\mathrm{cm}]$" % win_s); ax.legend()
    return _save(fig, out_dir, f"{prefix}_rolling_rms_e")


def plot_abs_e_distribution(res, prefix, out_dir):
    fig, ax = _new()
    q = np.linspace(0.0, 1.0, 500)
    for r in res:
        m = r["t"] >= CONFIG["t_from"]; col, ls = _style(r)
        ax.plot(np.quantile(100*np.abs(r["e"][m]), q), q, color=col, ls=ls, label=tex_label(r["label"]))
    ax.set_xlabel(r"$|e|\ [\mathrm{cm}]$"); ax.set_ylabel(r"$P(|e|\leq x)$"); ax.legend(loc="lower right")
    return _save(fig, out_dir, f"{prefix}_abs_e_distribution")


def plot_components(res, prefix, out_dir):
    fig, axs = _new((6.4, 5.4), 2, 1, sharex=True, layout="constrained")
    for r in res:
        t = r["t"]; ex, ey = _xy_err(r); m = t >= CONFIG["t_from"]; s = _stride(m.sum()); col, ls = _style(r)
        axs[0].plot(t[m][::s], ex[m][::s], color=col, ls=ls, label=tex_label(r["label"]))
        axs[1].plot(t[m][::s], ey[m][::s], color=col, ls=ls)
    axs[0].set_ylabel(r"$X_0-X_r\ [\mathrm{cm}]$"); axs[1].set_ylabel(r"$Y_0-Y_r\ [\mathrm{cm}]$")
    axs[1].set_xlabel(T_LABEL); axs[0].legend()
    return _save(fig, out_dir, f"{prefix}_cartesian_error_components")


def plot_speed(res, prefix, vref, out_dir):
    fig, ax = _new()
    for r in res:
        t = r["t"]; m = t >= CONFIG["t_from"]; s = _stride(m.sum()); col, ls = _style(r)
        ax.plot(t[m][::s], r["Y"][6][m][::s], color=col, ls=ls, label=tex_label(r["label"]))
    ax.axhline(vref, color="gray", lw=0.9, ls="-.", label=r"$v_{ref}$")
    ax.set_xlabel(T_LABEL); ax.set_ylabel(r"$v_{x0}\ [\mathrm{m/s}]$"); ax.legend()
    return _save(fig, out_dir, f"{prefix}_speed_vx0")


def plot_gp_quality(r, prefix, out_dir):
    """True residual, GP mean and +-2 sigma band of the four residual outputs (2x2)."""
    log = r["log"]; n = log["u"].size; t = r["t"][:n]; m = t >= CONFIG["t_from"]; s = _stride(m.sum())
    outs = [(log["res_x"][2], log["mu"][2], log["std"][2], r"$\dot{v}_{y0}\ [\mathrm{m/s^2}]$"),
            (log["res_x"][3], log["mu"][3], log["std"][3], r"$\dot{r}_{0}\ [\mathrm{rad/s^2}]$"),
            (log["res_x"][4], log["mu"][4], log["std"][4], r"$\ddot{\phi}\ [\mathrm{rad/s^2}]$"),
            (log["res_v"], log["mu"][6], log["std"][6], r"$\dot{v}_{x0}\ [\mathrm{m/s^2}]$")]
    fig, axs = _new((9.2, 6.6), 2, 2, sharex=True, layout="constrained")
    for k, (ax, (truth, mu, sd, yl)) in enumerate(zip(axs.ravel(), outs)):
        ax.plot(t[m][::s], truth[m][::s], color="black", lw=0.7, label=r"true residual")
        ax.plot(t[m][::s], mu[m][::s], color="tab:blue", ls="--", lw=1.1, label=r"GP mean")
        ax.fill_between(t[m][::s], (mu - 2*sd)[m][::s], (mu + 2*sd)[m][::s], color="tab:blue", alpha=0.25, lw=0,
                        label=r"$\pm 2\sigma$")
        ax.set_ylabel(yl)
        if k >= 2:
            ax.set_xlabel(T_LABEL)
    axs[0, 0].legend(loc="upper right")
    return _save(fig, out_dir, f"{prefix}_gp_quality_2sigma")


def plot_rms_percent_bars(groups, out_dir):
    """groups: list of (group label, [(role, % of 'no GP', RMS e in cm), ...])."""
    fig, ax = _new((8.6, 4.4)); w = 0.2; seen = set(); pct = r"\%" if CONFIG["use_tex"] else "%"
    ymax = max(v for _, items in groups for _, v, _ in items)
    for i, (glabel, items) in enumerate(groups):
        offs = (np.arange(len(items)) - (len(items) - 1)/2.0)*w
        for (role, val, cm), o in zip(items, offs):
            ax.bar(i + o, val, w*0.92, color=ROLE_STYLE[role][0], label=None if role in seen else ROLE_TEXT[role])
            seen.add(role)
            ax.text(i + o, val + 2, "%.0f%s (%.1f cm)" % (val, pct, cm), ha="center", va="bottom", rotation=90,
                    fontsize=7.5)
    ax.axhline(100, color="black", lw=0.8, ls="--")
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel(r"RMS of $e$ [" + pct + r" of no GP]"); ax.set_ylim(0, max(150, 1.5*ymax))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4, frameon=False); ax.grid(True, axis="x", alpha=0)
    return _save(fig, out_dir, "RMS_e_percent_of_noGP")


def rms_percent_groups(scenario_runs):
    """Bar-chart data: unaware controller, speed loop on; % RMS e of 'no GP' (offline GP only for scenario 1)."""
    ids = {k.split(".")[0]: (k, v) for k, v in scenario_runs.items()}
    groups = []
    for key, glabel in [("1", r"$\infty$-shape"), ("2", r"U-shape"), ("3", r"S-shape"), ("4", r"straight line")]:
        if key not in ids:
            continue
        sc = ids[key][1]; res = scenario_results(sc) + sc.get("extra_results", []); base = _rms_e(res[0])
        vals = {}
        for r in res:
            if (not _controller_aware(r)) and _speed_loop_on(r):
                role = bar_role(case_role(r))
                if role == "offline" and key != "1":
                    continue
                vals[role] = (100*_rms_e(r)/base, 100*_rms_e(r))
        items = [(role,) + vals[role] for role in BAR_ORDER if role in vals]
        groups.append((glabel, items))
        print(f"{glabel:>14s}: " + ", ".join(f"{role} {v:.1f}% ({cm:.1f} cm)" for role, v, cm in items))
    return groups


def save_error_figures(scenario_runs, vx_ref, out_dir="figures_svg"):
    """S1 (infinity): error norm, rolling RMS, components, |e| distribution, GP quality;
    S2 (U-turn), S3 (S-shape): same four error plots; S4 (line): |e|, speed, components; RMS bar chart."""
    plt.rcParams["figure.max_open_warning"] = 0
    ids = {k.split(".")[0]: (k, v) for k, v in scenario_runs.items()}
    with plt.rc_context(pub_rc()):
        if "1" in ids:
            sc = ids["1"][1]; res = scenario_results(sc)
            plot_cart_norm(res, "S1_eight", out_dir); plot_rolling_rms(res, "S1_eight", sc["win_s"], out_dir)
            plot_components(res, "S1_eight", out_dir); plot_abs_e_distribution(res, "S1_eight", out_dir)
            gp_run = [r for r in res if case_role(r) == "online_vx"] or [r for r in res if case_role(r) == "online"]
            if gp_run:
                plot_gp_quality(gp_run[0], "S1_eight", out_dir)
        for key, prefix in [("2", "S2_uturn"), ("3", "S3_sshape")]:
            if key in ids:
                sc = ids[key][1]; res = scenario_results(sc)
                plot_cart_norm(res, prefix, out_dir); plot_rolling_rms(res, prefix, sc["win_s"], out_dir)
                plot_abs_e_distribution(res, prefix, out_dir); plot_components(res, prefix, out_dir)
        if "4" in ids:
            res = scenario_results(ids["4"][1])
            plot_abs_e(res, "S4_line", out_dir); plot_speed(res, "S4_line", vx_ref, out_dir)
            plot_components(res, "S4_line", out_dir)
        plot_rms_percent_bars(rms_percent_groups(scenario_runs), out_dir)


# =============================================================================
# HF-state figures: "no GP" vs the best online GP
# =============================================================================
def _state_role(r):
    return bar_role(case_role(r))


def _runs(r_no, r_gp):
    role_gp = _state_role(r_gp)
    return [(r_no, ROLE_STYLE["baseline"][0], ROLE_TEXT["baseline"]), (r_gp, ROLE_STYLE[role_gp][0], ROLE_TEXT[role_gp])]


def _four(prefix, name, panels, runs, out_dir):
    """panels: 4 tuples (callable r -> (t, y), ylabel) drawn as a 2x2 subplot."""
    fig, axs = plt.subplots(2, 2, figsize=(9.2, 6.6), layout="constrained")
    for ax, (getter, ylabel) in zip(axs.ravel(), panels):
        for r, col, text in runs:
            t, y = getter(r); m = t >= CONFIG["t_from"]; s = _stride(m.sum())
            ax.plot(t[m][::s], y[m][::s], color=col, label=text)
        ax.set_ylabel(ylabel)
    for ax in axs[1]:
        ax.set_xlabel(T_LABEL)
    axs[0, 0].legend(loc="best")
    return _save(fig, out_dir, f"{prefix}_{name}")


def plot_path_and_inputs(prefix, r_no, r_gp, out_dir, equal_xy=True):
    """x-y path | articulation angle ; steering | traction force."""
    runs = _runs(r_no, r_gp)
    fig, axs = plt.subplots(2, 2, figsize=(9.2, 6.6), layout="constrained")
    ax = axs[0, 0]
    for r, col, text in runs:
        s = _stride(r["t"].size); ax.plot(r["Y"][2][::s], r["Y"][3][::s], color=col, label=text)
    ax.set_xlabel(r"$x_0\ [\mathrm{m}]$"); ax.set_ylabel(r"$y_0\ [\mathrm{m}]$")
    ax.plot(r_no["sig"]["Xr"], r_no["sig"]["Yr"], color="black", ls="--", lw=1.5, label=r"reference")
    ax.plot(r_no["Y"][2][0], r_no["Y"][3][0], marker="o", markersize=7, markerfacecolor="white",
            markeredgecolor="black", markeredgewidth=1.5, linestyle="None", label="start")
    ax.plot(r_no["Y"][2][-1], r_no["Y"][3][-1], marker="s", markersize=7, markerfacecolor="tab:red",
            linestyle="None", label="no GP final")
    ax.plot(r_gp["Y"][2][-1], r_gp["Y"][3][-1], marker="*", markersize=8,
            markerfacecolor=ROLE_STYLE[_state_role(r_gp)][0], linestyle="None", label="GP final")
    if equal_xy:
        ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", ncol=2)
    for r, col, text in runs:
        t = r["t"]; log = r["log"]; n = log["u"].size; m = t >= CONFIG["t_from"]; s = _stride(m.sum())
        axs[0, 1].plot(t[m][::s], np.rad2deg(r["Y"][5] - r["Y"][4])[m][::s], color=col, label=text)
        mu = m[:n]; su = _stride(mu.sum())
        axs[1, 0].plot(t[:n][mu][::su], np.rad2deg(log["u"])[mu][::su], color=col, label=text)
        axs[1, 1].plot(t[:n][mu][::su], (log["F_xf"]/1e3)[mu][::su], color=col, label=text)
    axs[0, 1].set_ylabel(r"articulation angle $\phi=\psi_1-\psi_0\ [\mathrm{deg}]$")
    axs[1, 0].set_ylabel(r"steering angle $\delta\ [\mathrm{deg}]$")
    axs[1, 1].set_ylabel(r"traction force $F_{xf}\ [\mathrm{kN}]$")
    for ax in (axs[0, 1], axs[1, 0], axs[1, 1]):
        ax.set_xlabel(T_LABEL)
        ax.legend(loc="upper center", ncol=2)
    return _save(fig, out_dir, f"{prefix}_states_path_articulation_inputs")


def plot_kinematic_states(prefix, r_no, r_gp, out_dir):
    g = lambda row: (lambda r: (r["t"], r["Y"][row]))
    return _four(prefix, "states_kinematic", [(g(2), r"$x_0\ [\mathrm{m}]$"), (g(3), r"$y_0\ [\mathrm{m}]$"),
                                              (g(4), r"$\psi_0\ [\mathrm{rad}]$"), (g(5), r"$\psi_1\ [\mathrm{rad}]$")],
                 _runs(r_no, r_gp), out_dir)


def plot_dynamic_states(prefix, r_no, r_gp, out_dir):
    g = lambda row: (lambda r: (r["t"], r["Y"][row]))
    return _four(prefix, "states_dynamic", [(g(6), r"$v_{x0}\ [\mathrm{m/s}]$"), (g(7), r"$v_{y0}\ [\mathrm{m/s}]$"),
                                            (g(8), r"$r_{0}\ [\mathrm{rad/s}]$"), (g(9), r"$r_{1}\ [\mathrm{rad/s}]$")],
                 _runs(r_no, r_gp), out_dir)


def best_online_gp(res):
    """Online GP run (unaware controller, speed loop on) with the lowest RMS e."""
    cands = [r for r in res if (not _controller_aware(r)) and _speed_loop_on(r)
             and _state_role(r) in ("online", "online_vx")]
    return min(cands, key=_rms_e)


def save_state_figures(scenario_runs, out_dir="figures_svg_states",
                       layout=(("1", "S1_eight", True), ("2", "S2_uturn", True),
                               ("3", "S3_sshape", False), ("4", "S4_line", False))):
    """Three 2x2 figures per scenario: path/articulation/inputs, kinematic states, dynamic states."""
    plt.rcParams["figure.max_open_warning"] = 0
    ids = {k.split(".")[0]: (k, v) for k, v in scenario_runs.items()}
    with plt.rc_context(pub_rc()):
        for key, prefix, equal_xy in layout:
            if key not in ids:
                continue
            sc = ids[key][1]; res = scenario_results(sc) + sc.get("extra_results", [])
            r_no = [r for r in res if case_role(r) == "baseline"][0]
            r_gp = best_online_gp(res)
            print(f"{prefix}: no GP '{r_no['label']}' ({100*_rms_e(r_no):.1f} cm)  vs  "
                  f"best online GP '{r_gp['label']}' ({100*_rms_e(r_gp):.1f} cm)")
            plot_path_and_inputs(prefix, r_no, r_gp, out_dir, equal_xy)
            plot_kinematic_states(prefix, r_no, r_gp, out_dir)
            plot_dynamic_states(prefix, r_no, r_gp, out_dir)
