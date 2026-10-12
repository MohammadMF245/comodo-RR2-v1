"""
truck_trailer: tractor-semitrailer path following with command-filtered backstepping and
semi-parametric online sparse GP residual compensation.

Modules (each usable on its own):
    params            vehicle parameters, perturbed plant parameters
    tires             tire models (linear, saturated, Fiala, Pacejka, custom)
    hf_model          nonlinear high-fidelity plant  (TruckTrailerDynamics)
    lf_model          linear design model, coordinates, steady cornering
    strict_feedback   strict-feedback form of any SISO (A, B) pair
    paths             reference paths (figure eight, line, U-turn, S-turn, lane change, custom)
    road              road bank angle scenarios
    controller        backstepping + reference filter + speed controller design
    spgp              generic semi-parametric online sparse GP
    residual_learning residual definition, features, training data, GP banks
    gp_compensation   GP -> F_hat estimators
    integrators       RK4
    closed_loop       closed-loop simulation with online learning
    scenarios         scenario / case runner
    metrics           performance metrics and tables
    plotting          quick-look plots
    figures           publication SVG figures
"""
from .params import DEFAULT_PARAMS, default_params, make_hf_params
from .tires import TIRE_MODELS, get_tire_model, register_tire_model
from .hf_model import HF_STATE_NAMES, TruckTrailerDynamics
from .lf_model import (LF_STATE_NAMES, build_lf_design, controllability_matrix, controllability_rank,
                       hf_to_lf, lateral_state_space, lf_to_z, steady_state_lf, vehicle_trailer_matrices)
from .strict_feedback import strict_feedback_form
from .paths import make_path, path_from_segments, path_length, reference_path_geom
from .road import bank_window, constant_bank, make_road, phi_roll_zero, roll_disturbance
from .controller import build_ctrl_dict, closed_loop_eigs, design_controller, design_reference_filter
from .residual_learning import (GP_FEATURE_NAMES, GP_TARGET_NAMES, generate_residual_dataset,
                                make_feature_normalizer, normalize_features, residual_hf_to_lf,
                                train_residual_gp_banks)
from .scenarios import (DEFAULT_GP_OPTS, bank_cases, make_speed, run_cases, run_scenario, standard_cases,
                        unaware_extra_cases)
from .metrics import (performance_metrics, print_gp_reliability, print_performance_table, report,
                      tracking_summary)

__version__ = "0.1.0"
