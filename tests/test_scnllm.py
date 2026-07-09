"""Focused tests for the synthetic SCN reference implementation.

These assert the mechanism-level properties the paper's figures rely on:
a synchronized endogenous rhythm near 24.2 h, entrainment (and re-entrainment)
to an independent 24.0 h zeitgeber, the documented 16-D state allocation, and a
clean separation between the SCN arm and the random/heuristic active controls.
Nothing here measures believability, engagement, or retrieval quality.
"""

from __future__ import annotations

import numpy as np
import pytest

import dataclasses

from scnllm.bench import (
    control_rows,
    dynamics_row,
    module_flop_estimate,
    overhead_projection,
    reduction_discretization_error,
)
from scnllm.oscillator import OscillatorConfig, estimate_period_hours, simulate
from scnllm.prc import phase_error
from scnllm.state_vector import state_vector
from scnllm.zeitgeber import make_schedule

MASTER = 20260709
SHIFT_AT = 24 * 9


@pytest.fixture(scope="module")
def runs():
    schedule = make_schedule(shift_at_hours=SHIFT_AT, shift_hours=6.0, seed=MASTER)
    entrained = simulate(schedule, seed=MASTER + 1)
    free_schedule = make_schedule(shift_at_hours=None, shift_hours=0.0, seed=MASTER)
    free_run = simulate(free_schedule, OscillatorConfig(entrainment_gain=0.0), seed=MASTER + 1)
    dynamics = dynamics_row(free_run, entrained, SHIFT_AT)
    controls = {row["arm"]: row for row in control_rows(entrained, shift_at_hours=SHIFT_AT, seed=MASTER + 2)}
    return {"entrained": entrained, "free": free_run, "dynamics": dynamics, "controls": controls}


# --- state vector -----------------------------------------------------------

def test_state_vector_shape_and_ranges():
    s = state_vector(phase=1.3, phase_velocity=0.26, zeitgeber_magnitude=0.7)
    assert s.shape == (16,)
    # amplitude, alertness, focus, energy, adaptation are bounded to [0, 1].
    amplitude, alertness, focus, energy, adaptation = s[3], s[4], s[5], s[6], s[7]
    for value in (amplitude, alertness, focus, energy, adaptation):
        assert 0.0 <= value <= 1.0
    # sin/cos phase encoding lands on the unit circle.
    assert s[0] == pytest.approx(np.sin(1.3))
    assert s[1] == pytest.approx(np.cos(1.3))


def test_state_vector_zeitgeber_scales_sensitivities():
    zero = state_vector(0.5, 0.0, 0.0)
    full = state_vector(0.5, 0.0, 1.0)
    # The 8 sensitivity weights (indices 8..15) scale with |Z|.
    assert np.allclose(zero[8:], 0.0)
    assert np.all(full[8:] > 0.0)


# --- schedule / ground-truth independence -----------------------------------

def test_target_phase_is_independent_of_oscillator():
    """The ground-truth phase comes from the imposed schedule, never the module
    under test -- so it must be identical regardless of oscillator parameters."""
    schedule = make_schedule(shift_at_hours=SHIFT_AT, shift_hours=6.0, seed=MASTER)
    a = simulate(schedule, OscillatorConfig(mu=0.05), seed=MASTER + 1)
    b = simulate(schedule, OscillatorConfig(mu=0.2, entrainment_gain=0.4), seed=MASTER + 5)
    assert np.array_equal(a.target_phase, b.target_phase)


def test_target_phase_matches_analytic_clock():
    schedule = make_schedule(shift_at_hours=None, shift_hours=0.0, seed=MASTER)
    clock = schedule.time_hours % 24.0
    assert np.allclose(schedule.target_phase, (2 * np.pi * clock / 24.0) % (2 * np.pi))


# --- endogenous rhythm ------------------------------------------------------

def test_ensemble_synchronizes(runs):
    assert runs["dynamics"]["free_running_order_parameter"] > 0.9


def test_free_running_period_near_intrinsic(runs):
    period = runs["dynamics"]["free_running_period_hours"]
    assert period == pytest.approx(24.2, abs=0.3)


def test_free_running_period_helper_matches():
    schedule = make_schedule(shift_at_hours=None, shift_hours=0.0, seed=MASTER)
    free = simulate(schedule, OscillatorConfig(entrainment_gain=0.0), seed=MASTER + 1)
    settled = free.time_hours > 72
    assert estimate_period_hours(free.time_hours[settled], free.phase[settled]) == pytest.approx(24.2, abs=0.3)


# --- entrainment ------------------------------------------------------------

def test_entrainment_lock_residual_tightness(runs):
    d = runs["dynamics"]
    # Locking tightness (jitter about the stable angle) is a SEPARATE metric from
    # the raw phase error; it is tight here, but is not claimed to be the target.
    assert d["pre_shift_lock_residual_percent"] < 5.0
    assert d["post_shift_lock_residual_percent"] < 5.0


def test_entrainment_angle_is_not_antiphase(runs):
    # The angle of entrainment is a modest, physical phase lag -- NOT the ~180
    # anti-phase lock the phi*-drive produced. Guards against that regression.
    assert abs(runs["dynamics"]["entrainment_angle_degrees"]) < 90.0


def test_raw_phase_error_reported_and_bounded(runs):
    # Raw phase error is the primary metric and is reported honestly (it includes
    # the entrainment angle, so it exceeds 5% but is not a gross failure).
    assert 0.0 < runs["dynamics"]["post_shift_raw_phase_error_percent"] < 25.0


def test_reentrainment_after_shift_is_finite(runs):
    relock = runs["dynamics"]["reentrainment_time_hours"]
    assert np.isfinite(relock)
    assert 0.0 < relock < 120.0  # re-locks within a few days


def test_reduced_phase_tracks_full_model(runs):
    # Algorithm 1 (faithful cosine-PRC reduction) vs the full second-order model,
    # post-shift, reported RAW (no offset subtraction).
    assert runs["dynamics"]["post_shift_reduction_error_percent"] < 8.0


# --- active-control separation ----------------------------------------------

def test_control_arms_separate_after_shift(runs):
    c = runs["controls"]
    # SCN and heuristic are both structured (high peak cross-correlation);
    # random is not.
    assert c["scn"]["pre_shift_peak_correlation"] > 0.7
    assert c["heuristic"]["pre_shift_peak_correlation"] > 0.7
    assert c["random"]["pre_shift_peak_correlation"] < 0.3
    # The discriminator is re-locking: SCN's zeitgeber lag barely moves across the
    # shift, while the clock-fixed heuristic's lag shifts by ~6 h.
    assert c["scn"]["lag_shift_hours"] < 1.5
    assert c["heuristic"]["lag_shift_hours"] > 4.0


def test_scn_and_random_periodicity(runs):
    c = runs["controls"]
    assert c["scn"]["dominant_period_hours"] == pytest.approx(24.0, abs=1.0)
    # Random arm has no ~24 h structure.
    assert c["random"]["dominant_period_hours"] < 12.0


# --- reproducibility --------------------------------------------------------

def test_pipeline_same_process_repeatable():
    # Same-process, same-seed repeatability. Cross-environment output is only
    # reproducible within a numerical tolerance (adaptive-step ODE), not bit-exact.
    schedule = make_schedule(shift_at_hours=SHIFT_AT, shift_hours=6.0, seed=MASTER)
    a = simulate(schedule, seed=MASTER + 1)
    b = simulate(schedule, seed=MASTER + 1)
    assert np.allclose(a.phase, b.phase, atol=1e-9)


# --- design guards for the recalibrated model -------------------------------

def test_target_phase_is_label_not_dynamics_input():
    """The oscillator is driven by the observable zeitgeber magnitude Z only;
    target_phase (phi*) is the evaluation label and MUST NOT enter the dynamics.
    Guard: rotating target_phase leaves the phase evolution unchanged, so the
    entrainment result cannot be an artifact of feeding the ground truth in."""
    schedule = make_schedule(shift_at_hours=None, shift_hours=0.0, seed=MASTER)
    rotated = dataclasses.replace(schedule, target_phase=(schedule.target_phase + np.pi) % (2 * np.pi))
    a = simulate(schedule, seed=MASTER + 1)
    b = simulate(rotated, seed=MASTER + 1)
    assert np.allclose(a.phase, b.phase, atol=1e-9)


def test_reduction_discretization_error_is_small():
    """The RESULTS.md solver-jitter claim is reproducible: Euler vs RK45 of the
    same scalar Algorithm-1 ODE agree to well under the ~2.4% projection offset."""
    schedule = make_schedule(shift_at_hours=SHIFT_AT, shift_hours=6.0, seed=MASTER)
    err = reduction_discretization_error(schedule, OscillatorConfig())
    assert err["mean_percent"] < 0.1
    assert err["max_percent"] < 0.5


def test_irregular_regime_degrades_but_does_not_collapse(runs):
    """README section 4 second regime: entrainment under a noisy/irregular
    schedule still locks (residual finite and bounded), tested honestly."""
    schedule = make_schedule(shift_at_hours=SHIFT_AT, shift_hours=6.0, irregular=True, seed=MASTER)
    entrained = simulate(schedule, seed=MASTER + 1)
    d = dynamics_row(runs["free"], entrained, SHIFT_AT)
    assert np.isfinite(d["post_shift_lock_residual_percent"])
    assert d["post_shift_lock_residual_percent"] < 15.0


def test_overhead_projection_is_negligible_and_labelled():
    flops = module_flop_estimate()
    assert flops["total_flops_per_update"] > 0
    rows = overhead_projection(flops["total_flops_per_update"], updates_per_query=1.0, tokens_per_query=256)
    # Oscillator/state primitives are negligible vs a 7B/70B forward pass.
    for row in rows:
        assert 0.0 < row["projected_overhead_percent"] < 1.0
