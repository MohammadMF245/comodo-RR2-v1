"""
High-fidelity (HF) nonlinear tractor-semitrailer model with nonlinear tires, road bank and
path-relative error states.

HF state (10):  [e, psi_tilde, x0, y0, psi0, psi1, vx0, vy0, r0, r1]
    e         lateral path error
    psi_tilde heading error w.r.t. the path
    x0, y0    tractor position
    psi0/1    tractor / trailer yaw angle
    vx0, vy0  tractor velocities (body frame)
    r0, r1    tractor / trailer yaw rate
Inputs: steering delta, front traction force F_xf, road bank angle phi_roll,
        path curvature kappa and path heading psir (for the error states).
"""
import numpy as np

from .tires import get_tire_model

HF_STATE_NAMES = ["e", "psi_tilde", "x0", "y0", "psi0", "psi1", "vx0", "vy0", "r0", "r1"]


class TruckTrailerDynamics:
    def __init__(self, params, tire_model="fiala"):
        self.p = params
        self.tire_model = tire_model.lower() if isinstance(tire_model, str) else tire_model
        self._tire = get_tire_model(tire_model)

    # ------------------------------------------------------------------ tires
    def tire_force(self, alpha, C):
        return self._tire(alpha, C, self.p)

    def slip_angles(self, vx0, vy0, r0, vx1, vy1, r1, delta):
        p = self.p
        alpha_f = np.arctan2((vy0 + p['lf']*r0), (vx0 + p['v_eps'])) - delta
        alpha_r = np.arctan2((vy0 - p['lr']*r0), (vx0 + p['v_eps']))
        beta_r = np.arctan2((vy1 - p['l11']*r1), (vx1 + p['v_eps']))
        return alpha_f, alpha_r, beta_r

    # ------------------------------------------------------------------ dynamics
    def accelerations(self, state, delta, F_xf, phi_roll):
        """state = [vx0, vy0, r0, r1, psi0, psi1] -> ([vx0', vy0', r0', r1'], [Fyf, Fyr, Fy1])."""
        p = self.p
        vx0, vy0, r0, r1, psi0, psi1 = state
        phi = psi1 - psi0
        phi_dot = r1 - r0
        d = p['g']*np.sin(phi_roll)

        F_res = p['Cr0'] + p['Cr1']*vx0**2

        vx1 = np.cos(phi)*vx0 + np.sin(phi)*(vy0 - p['lh']*r0)
        vy1 = -np.sin(phi)*vx0 + np.cos(phi)*(vy0 - p['lh']*r0) - p['l10']*r1

        af, ar, br = self.slip_angles(vx0, vy0, r0, vx1, vy1, r1, delta)

        Fyf = self.tire_force(af, p['Cf'])
        Fyr = self.tire_force(ar, p['Cr'])
        Fy1 = self.tire_force(br, p['C1'])

        M = np.array([
            [p['M0'] + p['M1'], 0, 0, p['M1']*p['l10']*np.sin(phi)],
            [0, p['M0'] + p['M1'], -p['M1']*p['lh'], -p['M1']*p['l10']*np.cos(phi)],
            [0, -p['lh']*p['M1'], p['Iz0'] + (p['lh']**2)*p['M1'], p['M1']*p['lh']*p['l10']*np.cos(phi)],
            [p['M1']*p['l10']*np.sin(phi), -p['M1']*p['l10']*np.cos(phi),
             p['M1']*p['lh']*p['l10']*np.cos(phi), p['Iz1'] + (p['l10']**2)*p['M1']],
        ])

        F = np.array([
            (F_xf*np.cos(delta)
             - Fyf*np.sin(delta)
             - np.sin(phi)*Fy1
             + p['M0']*vy0*r0
             + p['M1']*np.sin(phi)*vx1*r1
             + p['M1']*np.cos(phi)*vy1*r1
             - p['M1']*phi_dot*(vy0 - p['lh']*r0)
             - F_res),

            (F_xf*np.sin(delta)
             + Fyf*np.cos(delta)
             + Fyr
             + Fy1*np.cos(phi)
             - p['M0']*vx0*r0
             - p['M1']*np.cos(phi)*vx1*r1
             + p['M1']*np.sin(phi)*vy1*r1
             + p['M1']*phi_dot*vx0),

            (p['lf']*(F_xf*np.sin(delta) + Fyf*np.cos(delta))
             - p['lr']*Fyr
             - p['lh']*np.cos(phi)*Fy1
             + p['lh']*p['M1']*np.cos(phi)*vx1*r1
             - p['lh']*p['M1']*np.sin(phi)*vy1*r1
             - p['lh']*p['M1']*phi_dot*vx0),

            (-p['l1']*Fy1
             + p['l10']*p['M1']*vx1*r1
             + p['l10']*p['M1']*phi_dot*(-np.cos(phi)*vx0 - np.sin(phi)*vy0 + np.sin(phi)*p['lh']*r0)),
        ])

        B1 = np.array([
            -p['M1']*np.sin(phi),
            p['M0'] + p['M1']*np.cos(phi),
            -p['lh']*np.cos(phi)*p['M1'],
            -p['l10']*p['M1'],
        ])

        acc = np.linalg.solve(M, F + B1*d)
        tire_forces = np.array([Fyf, Fyr, Fy1])
        return acc, tire_forces

    def full_dynamics(self, state, delta, F_xf, phi_roll, kappa, psir):
        """Full 10-state HF dynamics.

        state = [e, psi_tilde, x0, y0, psi0, psi1, vx0, vy0, r0, r1]
        returns (xdot (10,), tire_forces (3,))
        """
        e, psi_tilde, x0, y0, psi0, psi1, vx0, vy0, r0, r1 = state

        acc, tire_forces = self.accelerations([vx0, vy0, r0, r1, psi0, psi1], delta, F_xf, phi_roll)
        dvx0, dvy0, dr0, dr1 = acc

        dx0 = vx0*np.cos(psi0) - vy0*np.sin(psi0)
        dy0 = vx0*np.sin(psi0) + vy0*np.cos(psi0)
        dpsi0 = r0
        dpsi1 = r1

        dpsi_tilde = r0 - kappa*(vx0*np.cos(psi0 - psir) - vy0*np.sin(psi0 - psir))/(1 - kappa*e)
        de = vx0*np.sin(psi0 - psir) + vy0*np.cos(psi0 - psir)

        return np.array([de, dpsi_tilde, dx0, dy0, dpsi0, dpsi1, dvx0, dvy0, dr0, dr1]), tire_forces
