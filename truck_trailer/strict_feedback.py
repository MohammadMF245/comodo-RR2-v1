"""
Strict-feedback (controllable canonical) form of a SISO linear system with a scalar disturbance.
Generic: works for any controllable (A, B1) pair, not only the truck model.
"""
import numpy as np


def strict_feedback_form(A, B1, B2, tol=1e-12, xp=np):
    """
    Transform
        x' = A x + B1 u + B2 d
    into
        z_1' = z_2 + c_1 d
        ...
        z_{n-1}' = z_n + c_{n-1} d
        z_n' = a^T z + b u + c_n d
    using z = T x.

    A  : (n,n);  B1, B2 : (n,) or (n,1)   (SISO control input, scalar disturbance)

    Returns
        T, T_inv : transformation z = T x and its inverse
        a        : (n,) last-row coefficients
        b        : scalar (= 1 up to round-off)
        c        : (n,) disturbance coefficients  c = T B2
    """
    A = xp.asarray(A)
    B1 = xp.asarray(B1).reshape(-1)
    B2 = xp.asarray(B2).reshape(-1)
    n = A.shape[0]

    # 1. controllability matrix C = [B1, A B1, ..., A^(n-1) B1]
    C = xp.column_stack([xp.linalg.matrix_power(A, i) @ B1 for i in range(n)])
    rank = xp.linalg.matrix_rank(C, tol=tol)
    if rank != n:
        raise ValueError(f"(A,B1) is not controllable: rank = {rank}, n = {n}")

    # 2. q^T = last row of C^-1  (q^T A^k B1 = 0 for k < n-1, = 1 for k = n-1)
    q = xp.linalg.inv(C)[-1, :]

    # 3. T = [q^T; q^T A; ...; q^T A^(n-1)]
    T = xp.vstack([q @ xp.linalg.matrix_power(A, i) for i in range(n)])
    T_inv = xp.linalg.inv(T)

    # 4. control coefficient (T B1 = [0,...,0,1]^T)
    b = (T @ B1)[-1]

    # 5. disturbance coefficients
    c = T @ B2

    # 6. last row:  a^T = q^T A^n T^-1
    a = q @ xp.linalg.matrix_power(A, n) @ T_inv
    return T, T_inv, a, b, c
