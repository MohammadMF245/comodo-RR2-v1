"""
Road cross slope (bank angle) phi_roll.

A road is a dict  {"phi_roll_fn": f(t, x_HF, theta) -> phi_roll [rad], "g": 9.81, "feedforward": bool}
  feedforward=True : the controller knows phi_roll and compensates d_r = g sin(phi_roll)
  feedforward=False: phi_roll acts on the truck but is unknown to the controller

Sign convention of the HF model: phi_roll > 0 accelerates the vehicle towards +y of the vehicle
frame (to the left), i.e. d_r = g sin(phi_roll) enters vy0' like a lateral force.
"""
import numpy as np


def phi_roll_zero(t, x_HF, theta):
    """Flat road."""
    return 0.0


def constant_bank(bank_rad):
    """phi_roll(t, x, theta) = bank_rad."""
    return lambda t, x_HF, theta: bank_rad


def bank_window(bank_rad, t_on, t_off):
    """phi_roll = bank_rad for t_on < t < t_off, else 0 (e.g. a banked road section)."""
    return lambda t, x_HF, theta: bank_rad if t_on < t < t_off else 0.0


def roll_disturbance(phi_roll, g):
    """Disturbance of the LF design model: d_r = g sin(phi_roll)  (matches B2_big)."""
    return g*np.sin(phi_roll)


def make_road(bank_rad=0.0, feedforward=False, g=9.81, phi_roll_fn=None):
    """Road dict. Constant bank `bank_rad` unless a custom phi_roll_fn(t, x_HF, theta) is given."""
    fn = phi_roll_fn if phi_roll_fn is not None else (lambda t, x_HF, theta: bank_rad)
    return {"phi_roll_fn": fn, "g": g, "feedforward": feedforward}
