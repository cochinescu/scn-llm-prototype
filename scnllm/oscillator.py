"""Coupled Van der Pol oscillator network with a phase-locking zeitgeber input.

The endogenous rhythm is a synchronized ensemble of ``n_oscillators`` Van der Pol
units (paper Eq. vdp). Three modelling choices matter and are documented in
``results/RESULTS.md``:

* **Resistive (velocity) coupling** ``K*(mean(v) - v_i)``. Diffusive *position*
  coupling ``k*(mean(x) - x_i)`` does not synchronize identical near-harmonic
  oscillators (the coupling cancels in the mean and merely detunes each unit);
  resistive coupling does, giving an order parameter ``R -> 1`` and a stable
  free-running period near ``intrinsic_period_hours``.
* **Entrainment from the observable zeitgeber only.** The oscillator is driven by
  the mean-subtracted zeitgeber magnitude ``g*(Z(t) - mean(Z))`` -- the same
  ``Z(t)`` a real sensor pipeline would emit. It is **not** driven by the
  environmental phase ``phi*``: ``phi*`` (``schedule.target_phase``) is used
  *only* as the independent evaluation label, never as a dynamics input, so the
  module must infer timing from the zeitgeber's temporal structure and
  non-entrainment remains possible. The additive own-phase PRC of Eq. (prc),
  ``-A*sin(phi - phi0)`` driven by the raw (non-negative) ``Z``, does not
  frequency-lock across the 24.2 h -> 24.0 h detuning (the ``Z >= 0`` rectified DC
  term acts as a frequency shift, not restoring coupling); mean-subtracting ``Z``
  removes that rectification. This is an honest finding reported for the paper.
* **Faithful phase reduction (Algorithm 1).** The zmag force reduces to a cosine
  PRC ``dphi/dt = omega + kappa*(Z - mean(Z))*cos(phi)`` with the *derived*
  ``kappa = -g/(a*omega)`` (``a`` = limit-cycle amplitude). This reduced update is
  consistent with the full model by construction, so their agreement is reported
  raw, with no offset subtraction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .prc import phase_error
from .zeitgeber import ZeitgeberSchedule


@dataclass(frozen=True)
class OscillatorConfig:
    n_oscillators: int = 5
    intrinsic_period_hours: float = 24.2
    mu: float = 0.05
    coupling: float = 0.2          # resistive/velocity mean-field coupling K
    entrainment_gain: float = 0.2  # external zeitgeber drive gain g
    frequency_spread: float = 0.02  # +/- fractional spread of intrinsic frequencies
    solver_rtol: float = 1e-6
    solver_atol: float = 1e-8


@dataclass(frozen=True)
class SimulationResult:
    time_hours: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    phase: np.ndarray
    reduced_phase: np.ndarray
    zeitgeber: np.ndarray
    target_phase: np.ndarray
    order_parameter: np.ndarray


def _interpolate(time: float, grid: np.ndarray, values: np.ndarray) -> float:
    return float(np.interp(time, grid, values))


def _mean_phase(x: np.ndarray, velocity: np.ndarray, omega: float) -> float:
    # The paper's mean-field read-out arg(mean(x_i + i*x_dot_i/omega)), expressed
    # in the forward-clock orientation for human-readable reporting.
    return float((-np.angle(np.mean(x + 1j * velocity / omega))) % (2 * np.pi))


def _order_parameter(x: np.ndarray, velocity: np.ndarray, omega: float) -> float:
    # Kuramoto order parameter of the individual oscillator phases; R -> 1 when
    # the ensemble is phase-synchronized into a coherent clock.
    theta = -np.angle(x + 1j * velocity / omega)
    return float(np.abs(np.mean(np.exp(1j * theta))))


def simulate(schedule: ZeitgeberSchedule, config: OscillatorConfig = OscillatorConfig(), seed: int = 11) -> SimulationResult:
    """Integrate the N coupled VdP reference model over a labelled schedule."""
    rng = np.random.default_rng(seed)
    t_eval = schedule.time_hours
    omega = 2 * np.pi / config.intrinsic_period_hours
    # Heterogeneous intrinsic frequencies so the resistive coupling does real
    # synchronizing work rather than trivially holding identical units together.
    omegas = omega * (1 + config.frequency_spread * rng.uniform(-1, 1, config.n_oscillators))
    phases = rng.uniform(0, 2 * np.pi, config.n_oscillators)
    initial = np.concatenate([np.cos(phases), -omega * np.sin(phases)])
    # Observable zeitgeber, mean-subtracted so the drive carries the daily
    # timing structure without a rectified DC frequency shift. target_phase is
    # NOT read here -- it is only an evaluation label (see module doc).
    z_centered = schedule.value - float(np.mean(schedule.value))

    def rhs(time: float, state: np.ndarray) -> np.ndarray:
        x, velocity = np.split(state, 2)
        z = _interpolate(time, t_eval, z_centered)
        # VdP dynamics with resistive mean-field coupling and an external drive
        # from the observable zeitgeber magnitude only (no phi*; see module doc).
        coupling = config.coupling * (float(np.mean(velocity)) - velocity)
        drive = config.entrainment_gain * z
        acceleration = config.mu * (1 - x**2) * velocity - omegas**2 * x + coupling + drive
        return np.concatenate([velocity, acceleration])

    solution = solve_ivp(
        rhs,
        (float(t_eval[0]), float(t_eval[-1])),
        initial,
        t_eval=t_eval,
        rtol=config.solver_rtol,
        atol=config.solver_atol,
        max_step=float(np.diff(t_eval).max()),
    )
    if not solution.success:
        raise RuntimeError(f"Oscillator integration failed: {solution.message}")
    x, velocity = np.split(solution.y, 2)
    phase = np.array([_mean_phase(x[:, i], velocity[:, i], omega) for i in range(x.shape[1])])
    order = np.array([_order_parameter(x[:, i], velocity[:, i], omega) for i in range(x.shape[1])])

    # Phase-reduced diagnostic (paper Algorithm 1): the FAITHFUL phase reduction
    # of the zmag full model is a cosine PRC with the derived coupling
    # kappa = -g/(a*omega), where a is the measured limit-cycle amplitude. Because
    # it is derived (not a different model), bench.dynamics_row scores its raw
    # agreement with the full model -- no offset subtraction.
    amplitude = float(np.median(np.sqrt(x**2 + (velocity / omega) ** 2)))
    kappa = -config.entrainment_gain / (amplitude * omega) if amplitude > 0 else 0.0
    reduced = np.empty_like(phase)
    reduced[0] = phase[0]
    for i in range(1, phase.size):
        dt = t_eval[i] - t_eval[i - 1]
        z = float(z_centered[i - 1])
        reduced[i] = (reduced[i - 1] + dt * (omega + kappa * z * np.cos(reduced[i - 1]))) % (2 * np.pi)
    return SimulationResult(t_eval, x, velocity, phase, reduced, schedule.value, schedule.target_phase, order)


def circular_phase_error_percent(phase: np.ndarray, target_phase: np.ndarray) -> float:
    """Raw mean absolute phase error vs the ground-truth phase, as a percentage of
    a 24-hour cycle. This is the primary, target-facing entrainment metric: it
    includes the (physical) angle of entrainment and is reported as-is, not
    reinterpreted away."""
    return float(np.mean(np.abs(phase_error(phase, target_phase))) / (2 * np.pi) * 100)


def entrainment_angle_radians(phase: np.ndarray, target_phase: np.ndarray) -> float:
    """Stable phase angle of entrainment: circular mean of (phase - target). A
    modest non-zero angle (a few hours' lag) is physically expected and is what
    makes the raw phase error larger than the locking tightness."""
    return float(np.angle(np.mean(np.exp(1j * phase_error(phase, target_phase)))))


def lock_residual_percent(phase: np.ndarray, target_phase: np.ndarray) -> float:
    """Phase-locking *tightness*: jitter of the phase-difference about its own
    stable angle, as a percentage of a cycle. This is a SEPARATE metric from the
    raw phase error above -- it measures how tightly the clock holds its lock, not
    how close it sits to the ground-truth phase. It is not a substitute for the
    raw error against the paper's target."""
    diff = phase_error(phase, target_phase)
    angle = np.angle(np.mean(np.exp(1j * diff)))
    return float(np.mean(np.abs(phase_error(diff, angle))) / (2 * np.pi) * 100)


def estimate_period_hours(time_hours: np.ndarray, phase: np.ndarray) -> float:
    """Estimate free-running period from a least-squares phase slope."""
    unwrapped = np.unwrap(phase)
    slope, _ = np.polyfit(time_hours, unwrapped, 1)
    return float(2 * np.pi / abs(slope))
