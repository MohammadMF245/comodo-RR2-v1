"""
Low-fidelity (LF) linear design model of the tractor-semitrailer.

4-state lateral dynamics (linear tires, constant speed V):
    M x4' = F x4 + D1 delta + D2 d_r,     x4 = [vy0, r0, phi_dot, phi]
augmented with the path-error kinematics to the 6-state design model
    x_LF = [e, psi_tilde, vy0, r0, phi_dot, phi]
    x_LF' = A_big x_LF + B1_big delta + B2_big d_r + B3_big d
    d   = kappa * vx0        (path curvature disturbance)
    d_r = g sin(phi_roll)    (road bank disturbance)

All functions take an array namespace `xp` (default NumPy). Pass `xp=jax.numpy`
to build the matrices with JAX (e.g. for differentiation w.r.t. parameters).
"""
import numpy as np

LF_STATE_NAMES = ["e", "psi_tilde", "vy0", "r0", "phi_dot", "phi"]


def vehicle_trailer_matrices(vx0, p, xp=np):
    """Matrices of the linear single-track tractor-trailer model at speed vx0.

    Returns
    -------
    M  : (4,4) mass / inertia matrix
    F  : (4,4) state matrix (before multiplication with M^-1)
    D1 : (4,1) steering input matrix
    D2 : (4,1) road-bank disturbance matrix
    Ks : understeer gradient
    """
    lf, lr, lh = p["lf"], p["lr"], p["lh"]
    l10, l11, l1 = p["l10"], p["l11"], p["l1"]
    M0, M1 = p["M0"], p["M1"]
    Iz0, Iz1 = p["Iz0"], p["Iz1"]
    Cf, Cr, C1 = p["Cf"], p["Cr"], p["C1"]
    vx = vx0 + p["v_eps"]

    M = xp.array([
        [M0 + M1, -M1*(l10 + lh), -M1*l10, 0.0],
        [-lh*M1, Iz0 + lh*M1*(l10 + lh), lh*M1*l10, 0.0],
        [-l10*M1, Iz1 + l10*M1*(l10 + lh), (Iz1 + M1*l10**2), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    F = xp.array([
        [-(Cf + Cr + C1)/vx,
         -(Cf*lf - Cr*lr - C1*(lh + l1))/vx - (M0 + M1)*vx,
         C1*l1/vx,
         C1],
        [-(lf*Cf - lr*Cr - lh*C1)/vx,
         -(lf**2*Cf + lr**2*Cr + lh*(lh + l1)*C1)/vx + lh*M1*vx,
         -lh*C1*l1/vx,
         -lh*C1],
        [l1*C1/vx,
         -l1*C1*(lh + l1)/vx + l10*M1*vx,
         -C1*l1**2/vx,
         -l1*C1],
        [0.0, 0.0, 1.0, 0.0],
    ])

    D2 = xp.array([[M0 + M1], [-lh*M1], [-l10*M1], [0.0]])
    D1 = xp.array([[Cf], [lf*Cf], [0.0], [0.0]])

    Ks = (M0*(Cr*lr - Cf*lf) - M1*l11*(Cr*(lh - lr) + Cf*(lh + lf)))/(Cf*Cr*(lf + lr))
    return M, F, D1, D2, Ks


def lateral_state_space(V, p, xp=np):
    """4-state lateral model x4' = A x4 + B1 delta + B2 d_r  at speed V. Returns A (4,4), B1 (4,1), B2 (4,1)."""
    M, F, D1, D2, _ = vehicle_trailer_matrices(V, p, xp)
    Mi = xp.linalg.inv(M)
    return Mi @ F, Mi @ D1, Mi @ D2


def augment_with_path_errors(A, B1, B2, V, xp=np):
    """6-state design model from the 4-state lateral model (adds e' = V psi_tilde + vy0, psi_tilde' = r0 - d).

    Returns a dict with A_big (6,6) and flat vectors B1_big, B2_big, B3_big (6,)."""
    A_big = xp.zeros((6, 6))
    if xp is np:
        A_big[0, 1] = V; A_big[0, 2] = 1.0; A_big[1, 3] = 1.0; A_big[2:, 2:] = A
    else:                                      # immutable arrays (JAX)
        A_big = A_big.at[0, 1].set(V).at[0, 2].set(1.0).at[1, 3].set(1.0).at[2:, 2:].set(A)
    z2 = xp.zeros(2)
    return {"A_big": A_big,
            "B1_big": xp.concatenate([z2, xp.ravel(B1)]),
            "B2_big": xp.concatenate([z2, xp.ravel(B2)]),
            "B3_big": xp.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0])}


def build_lf_design(V, p, xp=np):
    """LF design model at speed V: dict(A_big, B1_big, B2_big, B3_big, V)."""
    A, B1, B2 = lateral_state_space(V, p, xp)
    lf = augment_with_path_errors(A, B1, B2, V, xp)
    lf["V"] = V
    return lf


def controllability_matrix(A, B, xp=np):
    """[B, AB, ..., A^(n-1) B] for B of shape (n,) or (n, m)."""
    B = B.reshape(-1, 1) if B.ndim == 1 else B
    cols, AkB = [], B
    for _ in range(A.shape[0]):
        cols.append(AkB)
        AkB = A @ AkB
    return xp.hstack(cols)


def controllability_rank(A, B, tol=None):
    return int(np.linalg.matrix_rank(np.asarray(controllability_matrix(np.asarray(A), np.asarray(B))), tol=tol))


# =============================================================================
# Coordinates
# =============================================================================
def hf_to_lf(x_HF):
    """HF state -> LF state [e, psi_tilde, vy0, r0, phi_dot, phi]."""
    e, psi_t, X0, Y0, psi0, psi1, vx0, vy0, r0, r1 = x_HF
    return np.array([e, psi_t, vy0, r0, r1 - r0, psi1 - psi0])


def lf_to_z(x_LF, T):
    """LF state -> strict-feedback coordinates z = T x_LF."""
    return T @ x_LF


# =============================================================================
# Steady cornering
# =============================================================================
def steady_state_lf(lf, d0, d_roll):
    """LF steady state with e = 0 for curvature disturbance d0 = kappa*V and bank d_roll.
    Returns x_LF (6,) and delta."""
    Mx = np.column_stack([lf["A_big"][:, 1:], lf["B1_big"]])
    rhs = -(lf["B3_big"]*d0 + lf["B2_big"]*d_roll)
    sol = np.linalg.lstsq(Mx, rhs, rcond=None)[0]
    return np.r_[0.0, sol[:5]], sol[5]


def make_manifold_fn(params_lf, step=0.25):
    """V -> LF design model at that speed (cached on a grid of `step` m/s)."""
    cache = {}

    def fn(V):
        key = round(V/step)*step
        if key not in cache:
            cache[key] = build_lf_design(key, params_lf)
        return cache[key]
    return fn
