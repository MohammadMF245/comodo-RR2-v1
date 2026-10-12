# Tractor-trailer path following: backstepping + online sparse GP

Python package `truck_trailer/` and the notebook `Tractor_Trailer_Experiments.ipynb`, which sets up and
runs all experiments.

<!-- BEGIN: reusable-result capability description -->
## For AI-assisted method selection

<!--
Maintained for the CoMoDO reusable result.
Template: comodo-reusable-result/templates/capability-section.md
This section is read by the `find-methods` skill to match this repository
against an external user's application. Keep it accurate; update it when
examples are added or removed.
-->

### Identity

- **One-line summary:** Learn, while the controller runs, the mismatch between a linear design model and the real dynamics, and compensate it inside a model-based steering controller. Shown on lateral path following of a tractor-semitrailer with an unknown road bank. (Command-filtered backstepping on a linear model in strict-feedback form, plus a semi-parametric online sparse Gaussian process that learns the model residual.)
- **Role:** method library. The generic parts are the strict-feedback transform for a SISO linear system ([truck_trailer/strict_feedback.py](truck_trailer/strict_feedback.py)) and the semi-parametric online sparse GP ([truck_trailer/spgp.py](truck_trailer/spgp.py)). The tractor-semitrailer is the worked example.
- **Maturity:** simulation only. No hardware interface exists in the repository.
- **Maintainer / lab:** Mohammad Mahmoudi Filabadi, D2Lab, Ghent University (UGent).
- **Licence / availability:** no licence file in the repository; the maintainer has not stated one. Contact the maintainer.

### Physical setup

n/a for hardware, because this is a method library and simulation only. For reference, the simulated system in the worked example is:

- **System under control:** tractor-semitrailer, nonlinear 10-state planar model with tractor and trailer yaw, articulation, and Fiala / Pacejka / linear / saturated tire models ([truck_trailer/hf_model.py](truck_trailer/hf_model.py), [truck_trailer/tires.py](truck_trailer/tires.py)). Nominal parameters: tractor 5760 kg, trailer 6665 kg, hitch-to-trailer-axle 10 m ([truck_trailer/params.py](truck_trailer/params.py)).
- **Actuation:** front steering angle δ [rad] (lateral loop); front traction force F_xf [N] (speed loop, feedback-linearising).
- **Sensing:** the controller uses the full plant state with no noise and no delay: lateral path error, heading error, tractor lateral velocity, tractor and trailer yaw rates, articulation angle and longitudinal speed. The online-learning target is a state derivative minus the linear model; in this repository it is taken directly from the simulator ([truck_trailer/closed_loop.py:173](truck_trailer/closed_loop.py#L173)).
- **Sample rates:** fixed-step RK4 at dt = 5 ms (200 Hz). GP evaluated every 4 steps (50 Hz). Online GP update every 20 steps (10 Hz). Values from the notebook and from `DEFAULT_GP_OPTS` in [truck_trailer/scenarios.py](truck_trailer/scenarios.py).
- **Hardware & fieldbus:** none.
- **Scale of the dynamics:** speed 10 m/s. Path radii from 40 m to several hundred metres. Road bank up to 0.1 rad. Nominal closed-loop poles have max Re = −0.80 and max |λ| = 54 (notebook output).

### Problems this repository can help with

- My vehicle drifts sideways on a banked or cross-sloped road and I have no sensor for the slope. Learn the resulting disturbance online and compensate it in the steering loop. (Online sparse GP residual learning; road bank treated as unknown, the "unaware" GP bank.)
- I want to learn the mismatch between my linear design model and the real system while the controller runs, and feed that correction into a model-based controller. (Semi-parametric online sparse GP: parametric basis prior plus sparse GP, recursive updates, fixed-size data pool.)
- I need automated lateral control or path following for an articulated vehicle such as a tractor-trailer. (Command-filtered backstepping on a linear single-track tractor-trailer model with stable-inversion output reference.)
- I want to turn a single-input linear model into a chain of integrators so that I can design a backstepping controller on it. (Strict-feedback / controllable canonical transform for a SISO (A, B) pair with a matched disturbance.)

### Examples

All examples are cells of one notebook, [Tractor_Trailer_Experiments.ipynb](Tractor_Trailer_Experiments.ipynb). They share one design model and one controller; only the plant, path, road bank and initial speed change. All plant variants use the Pacejka tire model.

| Example | What it demonstrates | Entry point | Sim | Hardware |
| --- | --- | --- | --- | --- |
| Design + baseline | LF design model, controllability check, strict-feedback form, backstepping design at 10 m/s; figure-eight tracking without GP (0.5 m initial offset) | notebook §1–2; `tt.design_controller` in [truck_trailer/controller.py](truck_trailer/controller.py) | yes | no |
| Residual dataset + GP banks | 1000-sample residual dataset sampled around steady cornering; training of the "aware" and "unaware" GP banks | notebook §3; [truck_trailer/residual_learning.py](truck_trailer/residual_learning.py) | yes | no |
| Scenario 1: figure eight, mass/inertia mismatch | Plant masses and yaw inertias scaled by 1.3 (tractor) and 1.7 (trailer) against the nominal design; no GP vs offline GP vs online GP | notebook §4, first cell | yes | no |
| Scenarios 2–3: U-turn (R = 60 m) and lane change (3.5 m in 40 m), unknown bank 0.1 rad | Online GP learns an unmeasured road bank; compared with no GP and with bank feedforward | notebook §4, second cell | yes | no |
| Scenario 4: straight line, bank 0.1 rad, start at 13 m/s (ref 10 m/s) | Bank plus speed transient; GP aware/unaware; speed loop on/off | notebook §4, second cell | yes | no |
| Publication figures | SVG error and state figures for the four scenarios | notebook §5; [truck_trailer/figures.py](truck_trailer/figures.py) → `figures_svg/`, `figures_svg_states/` | yes | no |

### Preconditions

- **Full state measured.** The full state of the high-fidelity model must be available: lateral path error, heading error, lateral velocity, tractor and trailer yaw rates, articulation angle and longitudinal speed.
- **Derivatives for online learning.** The learning target is the state derivative minus the linear-model prediction, so a real application must supply these derivatives. In this repository they come from the simulator. <!-- inferred, please verify -->
- **Single-input, controllable, minimum-phase model.** You need a linear design model with one control input that is controllable (`strict_feedback_form` raises an error otherwise, [truck_trailer/strict_feedback.py:36](truck_trailer/strict_feedback.py#L36)). The input-to-tracked-output map must be minimum phase with relative degree 2 (`design_reference_filter` raises an error otherwise, [truck_trailer/controller.py:79-84](truck_trailer/controller.py#L79-L84)).
- **One operating point.** The design is linearised at a single speed (10 m/s), and the GP is pre-trained on 9.5–10.5 m/s. To operate over a speed range, the linear-model parameters must be computed per operating-point interval and supplied to the controller. That scheduling is not implemented here; only the per-speed model builder exists (`make_manifold_fn` in [truck_trailer/lf_model.py](truck_trailer/lf_model.py)).
- **Initial residual dataset.** You need an initial residual dataset to pre-train the GP. Here it is generated from the high-fidelity plant model around the steady-cornering manifold (`generate_residual_dataset`).
- **Path information.** Path curvature and heading at the vehicle's projection point must be available to the controller.

### Not suitable when

- **The full state vector is not directly measurable.** Add a state estimator first; none is provided here. Robustness to estimated rather than true states has not been tested.
- **You need the GP's predictive uncertainty for safety monitoring or chance constraints.** Don't rely on it. In the transient scenarios the variance is badly over-confident: RMS z-score 3.6–10.6 against an ideal of 1 (notebook §4 output, `print_gp_reliability`).
- **The model has several control inputs, or cannot be brought into single-input strict-feedback form.** Examples are a steered trailer axle or differential braking. Only the SISO case is implemented.
- **The steering-to-path-error output is non-minimum-phase.** The stable-inversion reference filter rejects such designs ([truck_trailer/controller.py:83](truck_trailer/controller.py#L83)).

Not exercised anywhere in this repository, so neither a limit nor a capability: measurement noise, sensor delay, operation far from 10 m/s, and deviations larger than about 0.5 m from the path.

### Dependencies beyond this repository

No `requirements.txt` or `pyproject.toml` exists; the list below is read from the imports.

| Dependency | Why it is needed | Hard requirement? |
| --- | --- | --- |
| Python 3.13 | Notebook kernel version (3.13.3); older versions untested | yes (version untested below 3.13) |
| NumPy ≥ 2.0 | All numerics; `np.trapezoid` is used in [truck_trailer/paths.py](truck_trailer/paths.py) | yes |
| SciPy | Cholesky solves and L-BFGS-B hyper-parameter tuning in [truck_trailer/spgp.py](truck_trailer/spgp.py) | yes |
| scikit-learn | k-means for the RBF basis centres (`pretrain_basis`) | yes, when the parametric basis is used |
| Matplotlib | Plots and SVG figures | only for plotting |
| Jupyter | Runs the experiment notebook | only for the examples |
| JAX | Optional array backend for the LF model matrices (`xp=jax.numpy`) | no |

### Where to look deeper

- [Tractor_Trailer_Experiments.ipynb](Tractor_Trailer_Experiments.ipynb): all experiments, with stored outputs, performance tables and GP reliability tables.
- [truck_trailer/spgp.py](truck_trailer/spgp.py): the generic, model-independent semi-parametric online sparse GP (init, predict, recursive update, sparsification, marginal-likelihood tuning, banks). Usable outside the truck example.
- [truck_trailer/strict_feedback.py](truck_trailer/strict_feedback.py): generic SISO strict-feedback transform.
- [truck_trailer/controller.py](truck_trailer/controller.py): backstepping law, command filters, stable-inversion reference filter, speed controller.
- [truck_trailer/residual_learning.py](truck_trailer/residual_learning.py): residual definition, GP features and masking. The steering input is masked from the GP to avoid a u → F̂ → u loop.
- [truck_trailer/closed_loop.py](truck_trailer/closed_loop.py): the simulation loop with zero-order-hold GP evaluation, one-step-delayed input and low-pass-filtered compensation.
- Notebook results worth reading before adopting the method (RMS lateral error, from the stored outputs):
  - **Offline GP vs online GP.** With the offline-trained GP and no online updates, tracking was worse than with no GP in scenario 1 (15.7 vs 12.8 cm). With online updates it was 5.1 cm.
  - **Known bank vs online GP.** Where the road bank is known to the controller, plain feedforward did better than the online "unaware" GP: lane change 4.7 vs 10.3 cm, straight line 2.1 vs 13.6 cm. The U-turn was comparable: 4.5 vs 3.9 cm.

<!-- END: reusable-result capability description -->
