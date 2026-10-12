"""
Fixed-step Runge-Kutta 4 integrators (generic).

rk4_step / simulate_rk4 : f(t, y, *args) -> dy
rk4_step_aux            : f(t, y, *args) -> dict with key 'dy' (+ auxiliary signals);
                          returns the dict of the first stage (signals at (t, y)) and y_next.
"""
import numpy as np


def rk4_step(f, t, y, dt, args=()):
    k1 = f(t, y, *args)
    k2 = f(t + dt/2, y + dt/2*k1, *args)
    k3 = f(t + dt/2, y + dt/2*k2, *args)
    k4 = f(t + dt, y + dt*k3, *args)
    return y + dt/6*(k1 + 2*k2 + 2*k3 + k4)


def rk4_step_aux(f, t, y, dt, args=()):
    k1 = f(t, y, *args)
    k2 = f(t + dt/2, y + dt/2*k1['dy'], *args)
    k3 = f(t + dt/2, y + dt/2*k2['dy'], *args)
    k4 = f(t + dt, y + dt*k3['dy'], *args)
    return k1, y + dt/6*(k1['dy'] + 2*k2['dy'] + 2*k3['dy'] + k4['dy'])


def simulate_rk4(f, y0, t_grid, args=()):
    """Fixed-step RK4 on t_grid. Stops (and truncates) if the state becomes non-finite."""
    Y = np.zeros((len(y0), len(t_grid)))
    Y[:, 0] = y0
    for k in range(len(t_grid) - 1):
        dt = t_grid[k+1] - t_grid[k]
        Y[:, k+1] = rk4_step(f, t_grid[k], Y[:, k], dt, args)
        if not np.all(np.isfinite(Y[:, k+1])):
            print(f"WARNING: simulation diverged at t = {t_grid[k+1]:.2f} s")
            return t_grid[:k+1], Y[:, :k+1]
    return t_grid, Y
