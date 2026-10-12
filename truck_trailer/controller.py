"""
Lateral command-filtered backstepping controller (designed on the LF model in strict-feedback
coordinates z = T x_LF) and a feedback-linearising longitudinal speed controller.

Design model (z-coordinates, G_i = 1):
    z_i' = z_{i+1} + c_i d + cr_i d_r + F_i ,          i = 1..5
    z_6' = a^T z + b u + c_6 d + cr_6 d_r + F_6
    d   = kappa * vx0                 (path curvature, c  = T B3_big)
    d_r = g sin(phi_roll)             (road bank,      cr = T B2_big)
    F_i = model residuals (HF - LF), estimated by F_hat_i

Controller:
    alpha_1 = -F_hat_1 + xdot_{1,d} - C_1 e_1 - c_1 d - cr_1 d_r
    alpha_i = -F_hat_i + xdot_{i,d} - C_i e_i - v_{i-1} - c_i d - cr_i d_r     i = 2..5
    u       = (-F_hat_6 + xdot_{6,d} - C_6 e_6 - v_5 - a^T z - c_6 d - cr_6 d_r) / b
    xddot_{i,d} = -2 zeta wf xdot_{i,d} - wf^2 (x_{i,d} - alpha_{i-1})   i = 2..6
    xidot_i     = -C_i xi_i + (x_{i+1,d} - alpha_i) + xi_{i+1}          i = 1..5, xi_6 = 0

The controller is described by two dicts:
    ctrl : T, a, b, c, c_roll, C (gains), wf, zeta, LF (design model)
    ref  : output reference filter (A, B, C, zeros_ue, use_F_hat)
"""
import numpy as np

from .lf_model import build_lf_design, hf_to_lf, lf_to_z
from .strict_feedback import strict_feedback_form
from .paths import path_quantities
from .road import roll_disturbance


# =============================================================================
# Design
# =============================================================================
def build_ctrl_dict(T, a, b, c, lf, C, wf, zeta):
    """Controller dictionary for the design (LF) model `lf` (dict with A_big, B1_big, B2_big, B3_big)."""
    T = np.asarray(T, dtype=np.float64)
    lf = {k: (np.asarray(v, dtype=np.float64) if k != "V" else v) for k, v in lf.items()}
    return {"T": T, "a": np.asarray(a, dtype=np.float64).ravel(), "b": float(b),
            "c": np.asarray(c, dtype=np.float64).ravel(),                    # curvature d
            "c_roll": T @ lf["B2_big"].ravel(),                              # bank d_r
            "C": np.asarray(C, dtype=np.float64), "wf": float(wf), "zeta": float(zeta),
            "LF": lf}


def design_controller(params, V_design, C=(0.8, 1.1, 1.35, 2.0, 6.2, 18.0), wf=32.0, zeta=0.99,
                      use_F_hat=True, tol=1e-20, verbose=True):
    """Complete lateral controller design at speed V_design.

    1. LF design model (6 states)          2. controllability check
    3. strict-feedback form z = T x_LF     4. backstepping gains / command filter
    5. stable-inversion output reference filter
    Returns ctrl, ref, lf."""
    lf = build_lf_design(V_design, params)
    T, T_inv, a, b, c = strict_feedback_form(lf["A_big"], lf["B1_big"], lf["B3_big"], tol=tol)
    ctrl = build_ctrl_dict(T, a, b, c, lf, C, wf, zeta)
    ref = design_reference_filter(ctrl["T"], ctrl["c"])
    ref["use_F_hat"] = use_F_hat          # also invert F_hat_1..5 in x_{1,d} so that e -> 0 with residuals
    if verbose:
        eig = closed_loop_eigs(lf["A_big"], lf["B1_big"], ctrl)
        print(f"LF design at V = {V_design:g} m/s | zeros u->e: {np.round(ref['zeros_ue'], 3)}")
        print(f"nominal closed loop: max Re = {eig.real.max():.3f}, max|eig| = {abs(eig).max():.1f}")
    return ctrl, ref, lf


# =============================================================================
# Output reference x_{1,d}: stable inversion of the u -> e zero dynamics
# =============================================================================
def design_reference_filter(T, c):
    """x_{1,d}(t) such that e = t^T z stays zero (t = first row of T^{-1}).

    With g_j = c_j d + cr_j d_r + F_hat_j:  z_k = s^{k-1} Z_1 - sum_{j<k} s^{k-1-j} G_j, so e = 0 needs
        P(s) Z_1 = sum_j N_j(s) G_j,  P(s) = sum_{k=1..5} t_k s^{k-1},
        N_j(s) = sum_{k=j+1..5} t_k s^{k-1-j}      (t_6 = 0: relative degree u -> e is 2).
    Realised in observable canonical form (shared denominator, 5 inputs g_1..g_5):
        w' = A w + B g,   x_{1,d} = C w,   xdot_{1,d} = C (A w + B g).
    Roots of P are the transmission zeros u -> e; they must be in the LHP.
    """
    t_row = np.linalg.inv(T)[0, :]
    if abs(t_row[5]) > 1e-8*np.abs(t_row).max():
        raise ValueError("expected relative degree 2 from u to e (t_6 = 0)")
    P_desc = t_row[4::-1]                                   # [t5, t4, t3, t2, t1]
    zeros_ue = np.roots(P_desc)
    if np.any(zeros_ue.real >= 0):
        raise ValueError("non-minimum-phase output: stable inversion not possible")
    p_asc = t_row[:5]/t_row[4]                              # [p0..p3, 1], monic
    A = np.zeros((4, 4)); A[1:, :3] = np.eye(3); A[:, 3] = -p_asc[:4]
    B = np.zeros((4, 5))
    for j in range(5):                                      # input g_{j+1}
        for k in range(j + 1, 5):                           # t_{k+1}, power k-1-j
            B[k - 1 - j, j] += t_row[k]/t_row[4]
    C = np.array([0.0, 0.0, 0.0, 1.0])
    return {"A": A, "B": B, "C": C, "zeros_ue": zeros_ue}


def reference_filter_output(w, g, ref):
    """x_{1,d} and its exact derivative from filter state w and input g (shape (5,))."""
    return ref["C"] @ w, ref["C"] @ (ref["A"] @ w + ref["B"] @ g)


def reference_filter_rhs(w, g, ref):
    return ref["A"] @ w + ref["B"] @ g


def reference_filter_input(d0, d_roll, F_hat, c, c_roll, use_F_hat=True):
    """g_j = c_j d + cr_j d_r + F_hat_j, j = 1..5."""
    return c[:5]*d0 + c_roll[:5]*d_roll + (F_hat[:5] if use_F_hat else 0.0)


# =============================================================================
# Lateral controller (command-filtered backstepping)
# =============================================================================
def lateral_backstepping_controller(z, xd, xdd, xi, d0, d_roll, x1d, x1d_dot, F_hat,
                                    C, a, b, c, c_roll):
    """Virtual controls alpha_1..alpha_6 (alpha_6 = u = steering), errors e, v.
    xd, xdd: x_{i,d}, xdot_{i,d} with index i = 2..6 (length-7 arrays, entries 0,1 unused)
    xi:      compensation states with index i = 1..5 (length-7 array, xi_6 = 0)."""
    alpha = np.zeros(7); e = np.zeros(7); v = np.zeros(7)
    e[1] = z[0] - x1d
    alpha[1] = -F_hat[0] + x1d_dot - C[0]*e[1] - c[0]*d0 - c_roll[0]*d_roll
    for i in range(2, 7):
        e[i] = z[i-1] - xd[i]
        v[i-1] = e[i-1] - xi[i-1]
        alpha[i] = -F_hat[i-1] + xdd[i] - C[i-1]*e[i] - v[i-1] - c[i-1]*d0 - c_roll[i-1]*d_roll
    alpha[6] = (alpha[6] - a @ z)/b                         # last step: cancel a^T z, divide by b
    return alpha, e, v


def command_filter_rhs(xd, xdd, alpha, wf, zeta):
    """Second-order command filters for i = 2..6. Returns (xd', xdd') for i = 2..6."""
    return xdd[2:7].copy(), -2.0*zeta*wf*xdd[2:7] - wf**2*(xd[2:7] - alpha[1:6])


def compensation_rhs(xd, xi, alpha, C):
    """Compensation signals for i = 1..5 (G_i = 1, xi_6 = 0)."""
    return np.array([-C[i-1]*xi[i] + (xd[i+1] - alpha[i]) + xi[i+1] for i in range(1, 6)])


# =============================================================================
# Longitudinal controller (F_xf)
# =============================================================================
def longitudinal_controller(x_HF, vx_ref, vx_dot_ref, k_vx, params_ctrl, res_vx_hat=0.0):
    """Feedback-linearising traction force for the speed loop, using the controller's
    (nominal) parameters params_ctrl. res_vx_hat: learned residual of vx0' (compensated)."""
    e, psi_t, X0, Y0, psi0, psi1, vx0, vy0, r0, r1 = x_HF
    vy1_est = vy0 - params_ctrl['lh']*r0 - params_ctrl['l10']*r1 - vx0*(psi1 - psi0)
    return (-params_ctrl['M0']*vy0*r0 - params_ctrl['M1']*vy1_est*r1
            + (params_ctrl['M0'] + params_ctrl['M1'])*(-k_vx*(vx0 - vx_ref) + vx_dot_ref - res_vx_hat))


# =============================================================================
# Initialisation and linear check
# =============================================================================
def initial_controller_states(t0, x0_HF, theta0, ctrl, ref, path, road, F_hat0):
    """Controller initial states:
       w(0)       : reference filter at steady state for g(0)
       xi(0) = 0, xdot_{i,d}(0) = 0, x_{i,d}(0) = alpha_{i-1}(0).
    F_hat0: residual estimate (z-coordinates) at t0."""
    _, _, _, _, d0, _ = path_quantities(x0_HF, theta0, path)
    z0 = lf_to_z(hf_to_lf(x0_HF), ctrl["T"])
    F_hat0 = np.asarray(F_hat0, dtype=float)
    phi_roll0 = float(road["phi_roll_fn"](t0, x0_HF, theta0))
    d_roll0 = roll_disturbance(phi_roll0, road["g"]) if road["feedforward"] else 0.0
    g0 = reference_filter_input(d0, d_roll0, F_hat0, ctrl["c"], ctrl["c_roll"], ref["use_F_hat"])
    w0 = -np.linalg.solve(ref["A"], ref["B"] @ g0)
    x1d0, x1d_dot0 = reference_filter_output(w0, g0, ref)

    # alpha_{i-1} depends on x_{i-1,d} only through e_{i-1}; build x_{i,d}(0) sequentially
    C, c, c_roll = ctrl["C"], ctrl["c"], ctrl["c_roll"]
    xd0 = np.zeros(7); xdd0 = np.zeros(7); xi0 = np.zeros(7)
    e_prev = z0[0] - x1d0
    alpha_prev = -F_hat0[0] + x1d_dot0 - C[0]*e_prev - c[0]*d0 - c_roll[0]*d_roll0
    for i in range(2, 7):
        xd0[i] = alpha_prev
        e_i = z0[i-1] - xd0[i]
        alpha_prev = (-F_hat0[i-1] - C[i-1]*e_i - e_prev - c[i-1]*d0
                      - c_roll[i-1]*d_roll0)                  # xdd=0, xi=0 -> v = e
        e_prev = e_i
    return xd0, xdd0, xi0, w0


def closed_loop_eigs(A_plant, B_plant, ctrl):
    """Eigenvalues of [x_LF, xd, xdd, xi] closed loop for x' = A x + B u, d = 0, x1d = 0."""
    A_plant = np.asarray(A_plant, float)
    B_plant = np.asarray(B_plant, float).ravel()

    def f(yv):
        x = yv[:6]
        xd = np.zeros(7); xdd = np.zeros(7); xi = np.zeros(7)
        xd[2:7] = yv[6:11]; xdd[2:7] = yv[11:16]; xi[1:6] = yv[16:21]
        al, _, _ = lateral_backstepping_controller(ctrl["T"] @ x, xd, xdd, xi, 0.0, 0.0, 0.0, 0.0,
                                                   np.zeros(6), ctrl["C"], ctrl["a"], ctrl["b"],
                                                   ctrl["c"], ctrl["c_roll"])
        dxd, dxdd = command_filter_rhs(xd, xdd, al, ctrl["wf"], ctrl["zeta"])
        return np.concatenate([A_plant @ x + B_plant*al[6], dxd, dxdd,
                               compensation_rhs(xd, xi, al, ctrl["C"])])
    return np.linalg.eigvals(np.column_stack([f(col) for col in np.eye(21)]))
