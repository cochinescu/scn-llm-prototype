"""Small measurement harness for the four prototype figures."""

from __future__ import annotations

import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import periodogram

from .controls import heuristic_modulation, random_modulation, scn_modulation
from .oscillator import (
    OscillatorConfig,
    SimulationResult,
    circular_phase_error_percent,
    entrainment_angle_radians,
    estimate_period_hours,
    lock_residual_percent,
    simulate,
)
from .prc import phase_error
from .rag_modulation import make_toy_corpus, retrieve, retrieval_parameters
from .state_vector import state_vector
from .zeitgeber import ZeitgeberSchedule


def confidence_interval_ms(samples_seconds: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(samples_seconds) * 1_000
    median = float(np.median(values))
    lower, upper = np.percentile(values, [2.5, 97.5])
    return median, float(lower), float(upper)


def time_callable(callable_, repeats: int = 160, warmup: int = 20) -> tuple[float, float, float]:
    for _ in range(warmup):
        callable_()
    samples = np.empty(repeats)
    for i in range(repeats):
        start = time.perf_counter()
        callable_()
        samples[i] = time.perf_counter() - start
    return confidence_interval_ms(samples)


def _algorithm1_update(phi: float, velocity: float, z_centered: float, dt: float, omega: float, kappa: float) -> np.ndarray:
    """One genuine inference-time temporal update: the Algorithm-1 cosine-PRC
    phase advance followed by the 16-D state map. This is the real per-update
    primitive cost (not a stubbed constant-velocity add)."""
    phi_next = (phi + dt * (omega + kappa * z_centered * np.cos(phi))) % (2 * np.pi)
    return state_vector(phi_next, velocity, abs(z_centered))


def latency_rows(result: SimulationResult, config: OscillatorConfig = OscillatorConfig(), corpus_sizes: tuple[int, ...] = (128, 512, 2048)) -> list[dict[str, float | int | str]]:
    dt = float(np.median(np.diff(result.time_hours)))
    velocity = float(np.median(np.gradient(np.unwrap(result.phase), result.time_hours)))
    omega = 2 * np.pi / config.intrinsic_period_hours
    amplitude = float(np.median(np.sqrt(result.positions ** 2 + (result.velocities / omega) ** 2)))
    kappa = -config.entrainment_gain / (amplitude * omega) if amplitude > 0 else 0.0
    z_centered = float(result.zeitgeber[-1] - np.mean(result.zeitgeber))
    state = state_vector(float(result.phase[-1]), velocity, float(result.zeitgeber[-1]))
    params = retrieval_parameters(state)
    rows: list[dict[str, float | int | str]] = []
    rows.append(_latency_row("state_vector", 0, lambda: state_vector(float(result.phase[-1]), velocity, float(result.zeitgeber[-1]))))
    # A genuine Algorithm-1 temporal update: cosine-PRC phase advance + state map.
    rows.append(_latency_row("scn_update", 0, lambda: _algorithm1_update(float(result.phase[-1]), velocity, z_centered, dt, omega, kappa)))
    for size in corpus_sizes:
        embeddings, timestamps = make_toy_corpus(size=size)
        query = embeddings[0].copy()
        rows.append(_latency_row("retrieval_baseline", size, lambda e=embeddings, t=timestamps: retrieve(query, e, t, result.time_hours[-1])))
        rows.append(_latency_row("retrieval_modulated", size, lambda e=embeddings, t=timestamps: retrieve(query, e, t, result.time_hours[-1], params)))
    return rows


def _latency_row(component: str, corpus_size: int, callback) -> dict[str, float | int | str]:
    median, lower, upper = time_callable(callback)
    return {"component": component, "corpus_size": corpus_size, "median_ms": median, "ci_low_ms": lower, "ci_high_ms": upper}


def reentrainment_time_hours(entrained: SimulationResult, shift_at_hours: float, tolerance_frac: float = 0.10) -> float:
    """Hours after the schedule shift until the phase-difference settles to
    within ``tolerance_frac`` of a cycle of its post-shift stable angle and stays
    there. Returns NaN if it never re-locks within the run."""
    time = entrained.time_hours
    diff = phase_error(entrained.phase, entrained.target_phase)
    post = time >= shift_at_hours
    settled = time >= (shift_at_hours + (time[-1] - shift_at_hours) / 2)
    angle = np.angle(np.mean(np.exp(1j * diff[settled])))
    deviation = np.abs(phase_error(diff, angle))
    threshold = tolerance_frac * 2 * np.pi
    for k in np.where(post)[0]:
        if np.all(deviation[k:] < threshold):
            return float(time[k] - shift_at_hours)
    return float("nan")


def dynamics_row(free_run: SimulationResult, entrained: SimulationResult, shift_at_hours: float) -> dict[str, float]:
    """Endogenous-rhythm and entrainment diagnostics.

    The **raw phase error** vs the independent ground truth is the primary,
    target-facing metric and is reported as-is (it includes the physical angle of
    entrainment). The **lock residual** is reported separately as a distinct
    locking-tightness measure, not a substitute for the raw error. The full-vs-
    reduced agreement is reported RAW (no offset subtraction): the reduced
    Algorithm-1 model is the faithful cosine-PRC reduction of the full model, so
    they are consistent by construction.
    """
    time = entrained.time_hours
    post = time >= shift_at_hours
    pre = ~post
    # Skip the initial synchronization transient (~3 days) for the period fit.
    settled_free = free_run.time_hours > 72
    return {
        "free_running_period_hours": estimate_period_hours(free_run.time_hours[settled_free], free_run.phase[settled_free]),
        "free_running_order_parameter": float(np.mean(free_run.order_parameter[settled_free])),
        "entrainment_angle_degrees": float(np.degrees(entrainment_angle_radians(entrained.phase[post], entrained.target_phase[post]))),
        "pre_shift_raw_phase_error_percent": circular_phase_error_percent(entrained.phase[pre], entrained.target_phase[pre]),
        "post_shift_raw_phase_error_percent": circular_phase_error_percent(entrained.phase[post], entrained.target_phase[post]),
        "pre_shift_lock_residual_percent": lock_residual_percent(entrained.phase[pre], entrained.target_phase[pre]),
        "post_shift_lock_residual_percent": lock_residual_percent(entrained.phase[post], entrained.target_phase[post]),
        "reentrainment_time_hours": reentrainment_time_hours(entrained, shift_at_hours),
        "post_shift_reduction_error_percent": float(np.mean(np.abs(phase_error(entrained.phase[post], entrained.reduced_phase[post]))) / (2 * np.pi) * 100),
    }


def reduction_discretization_error(schedule: ZeitgeberSchedule, config: OscillatorConfig) -> dict[str, float]:
    """Quantify the forward-Euler discretization error of the Algorithm-1
    phase-reduced update against a high-accuracy RK45 integration of the *same*
    scalar cosine-PRC ODE (``dphi/dt = omega + kappa*(Z-mean)*cos(phi)``). This
    isolates solver jitter, so the reduced-vs-full error is attributed to the
    phase reduction, not the integrator. The coupling ``kappa`` is taken from a
    full run so the diagnostic matches the reduced model actually used."""
    full = simulate(schedule, config)
    t = schedule.time_hours
    omega = 2 * np.pi / config.intrinsic_period_hours
    amplitude = float(np.median(np.sqrt(full.positions ** 2 + (full.velocities / omega) ** 2)))
    kappa = -config.entrainment_gain / (amplitude * omega) if amplitude > 0 else 0.0
    z_centered = schedule.value - float(np.mean(schedule.value))
    phi0 = float(full.phase[0])

    euler = np.empty_like(t)
    euler[0] = phi0
    for i in range(1, t.size):
        dt = t[i] - t[i - 1]
        euler[i] = (euler[i - 1] + dt * (omega + kappa * z_centered[i - 1] * np.cos(euler[i - 1]))) % (2 * np.pi)

    def rhs(tt: float, phi: np.ndarray) -> np.ndarray:
        z = np.interp(tt, t, z_centered)
        return omega + kappa * z * np.cos(phi)

    solution = solve_ivp(rhs, (float(t[0]), float(t[-1])), [phi0], t_eval=t, rtol=1e-9, atol=1e-12, max_step=float(np.diff(t).max()))
    rk45 = solution.y[0] % (2 * np.pi)
    err = np.abs(phase_error(euler, rk45)) / (2 * np.pi) * 100
    return {"mean_percent": float(np.mean(err)), "max_percent": float(np.max(err))}


# --- Figure 1: analytic op-count and a labelled system-overhead projection ----

def module_flop_estimate() -> dict[str, int]:
    """Analytic per-update floating-point op-count for the temporal module's
    inference-time primitives (Algorithm-1 phase update + the 16-D state map).
    Costs are counted at ~10 ops per transcendental; this is an estimate, not a
    hardware measurement, and covers only the oscillator/state primitives -- NOT
    the sensor-ingestion or RAG-modulation machinery the paper's system-level
    budget attributes most of its cost to."""
    trig = 10  # nominal cost of one sin/cos
    # Algorithm-1 phase update: omega + kappa*z*cos(phi), Euler step. One cos plus
    # six non-trig ops: kappa*z, *cos, omega+, dt*, phi+, wrap.
    phase_update = trig + 6
    # State map: sin, cos, 3 cos (alertness/focus/energy), amplitude clip,
    # adaptation, 8 sensitivity multiplies.
    state_map = 2 * trig + 3 * trig + 8 + 6
    return {
        "phase_update_flops": phase_update,
        "state_map_flops": state_map,
        "total_flops_per_update": phase_update + state_map,
    }


def overhead_projection(
    module_flops_per_update: int,
    updates_per_query: float,
    tokens_per_query: int,
    model_params: tuple[int, ...] = (7_000_000_000, 70_000_000_000),
) -> list[dict[str, float]]:
    """Derive a labelled system-overhead PROJECTION from the measured/estimated
    module cost against published forward-pass FLOP figures, under an explicitly
    stated denominator. A dense forward pass costs ~2*N_params FLOPs per token.
    The projection is (module FLOPs per query) / (model FLOPs per query); it is a
    projection under stated assumptions, never a measured system overhead, and it
    deliberately does not carry the paper's 15% target forward."""
    module_per_query = module_flops_per_update * updates_per_query
    rows = []
    for params in model_params:
        model_per_query = 2 * params * tokens_per_query
        rows.append({
            "model_params_billions": params / 1e9,
            "model_flops_per_query": float(model_per_query),
            "module_flops_per_query": float(module_per_query),
            "projected_overhead_percent": module_per_query / model_per_query * 100,
        })
    return rows


def control_rows(result: SimulationResult, shift_at_hours: float | None = None, seed: int = 31) -> list[dict[str, float | str]]:
    """Compare the three modulation arms (the paper's active-control design).

    The heuristic arm is a fixed *absolute-clock* lookup (condition 3): it is tied
    to wall-clock time-of-day, not to the user's schedule. When the user's routine
    shifts (the 6 h schedule shift), the SCN arm re-entrains and its correlation
    with the shifted zeitgeber recovers, whereas the clock-fixed heuristic does not
    follow the shift. That post-shift separation -- not the raw periodogram, which
    a 24 h step function also passes -- is the honest SCN-vs-lookup distinction.
    """
    sample_hours = float(np.median(np.diff(result.time_hours)))
    absolute_clock_phase = 2 * np.pi * (result.time_hours % 24.0) / 24.0
    arms = {
        "scn": scn_modulation(result.phase),
        "random": random_modulation(result.phase.size, seed=seed),
        "heuristic": heuristic_modulation(absolute_clock_phase),
    }
    zeitgeber = result.zeitgeber - np.mean(result.zeitgeber)
    if shift_at_hours is None:
        pre_mask = np.ones(result.time_hours.size, dtype=bool)
        post_mask = pre_mask
    else:
        post_mask = result.time_hours >= shift_at_hours
        pre_mask = ~post_mask

    def _pearson(x: np.ndarray, y: np.ndarray) -> float:
        x = x - x.mean()
        y = y - y.mean()
        denom = np.sqrt(float(np.dot(x, x)) * float(np.dot(y, y)))
        return float(np.dot(x, y) / denom) if denom > 0 else 0.0

    def best_lag(signal: np.ndarray, ref: np.ndarray, max_lag_hours: float = 12.0) -> tuple[float, float]:
        """Peak absolute cross-correlation with the zeitgeber and the lag (hours)
        at which it occurs. Phase-invariant: unaffected by the arbitrary phase
        offset of a modulation convention, so it measures alignment honestly."""
        if signal.std() == 0 or ref.std() == 0:
            return 0.0, 0.0
        max_lag = int(max_lag_hours / sample_hours)
        best_c, best_l = 0.0, 0
        # Scan by ascending |lag| so that when the peak |corr| ties between the
        # true alignment and the anti-phase point (12 h away on a ~24 h signal),
        # the smaller-magnitude (true-alignment) lag wins instead of whichever
        # end of the range is visited first.
        for lag in sorted(range(-max_lag, max_lag + 1), key=abs):
            aa, bb = (signal[-lag:], ref[: signal.size + lag]) if lag < 0 else (signal[: signal.size - lag], ref[lag:])
            if aa.size < signal.size // 2:
                continue
            c = _pearson(aa, bb)
            if abs(c) > abs(best_c):
                best_c, best_l = c, lag
        return abs(best_c), best_l * sample_hours

    rows = []
    for name, signal in arms.items():
        centered = signal - np.mean(signal)
        frequencies, power = periodogram(centered, fs=1 / sample_hours)
        valid = frequencies > 0
        peak_frequency = float(frequencies[valid][np.argmax(power[valid])])
        pre_corr, pre_lag = best_lag(signal[pre_mask], zeitgeber[pre_mask])
        post_corr, post_lag = best_lag(signal[post_mask], zeitgeber[post_mask])
        # Wrap the lag change to [-12, 12] h. SCN re-entrains (~0 h); the
        # clock-fixed heuristic does not follow the 6 h shift (~6 h).
        lag_shift = ((post_lag - pre_lag + 12.0) % 24.0) - 12.0
        rows.append({
            "arm": name,
            "dominant_period_hours": 1 / peak_frequency,
            "pre_shift_peak_correlation": pre_corr,
            "post_shift_peak_correlation": post_corr,
            "lag_shift_hours": abs(lag_shift),
            "modulation_std": float(np.std(signal)),
        })
    return rows
