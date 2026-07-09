"""Fixed analytic first-cut state map, with the documented 16-D allocation."""

from __future__ import annotations

import numpy as np


def state_vector(phase: float, phase_velocity: float, zeitgeber_magnitude: float) -> np.ndarray:
    """Map (phase, phase velocity, |Z|) to a documented 16-D temporal state.

    This is intentionally analytic rather than trained.  It supports cost and
    latency measurements without representing a trained behavioural model.
    """
    phase_unit = (phase % (2 * np.pi)) / (2 * np.pi)
    amplitude = float(np.clip(zeitgeber_magnitude, 0.0, 1.0))
    alertness = 0.5 + 0.5 * np.cos(phase - np.pi)
    focus = 0.5 + 0.5 * np.cos(phase - 1.2 * np.pi)
    energy = 0.5 + 0.5 * np.cos(phase - 0.9 * np.pi)
    adaptation = 0.25 + 0.75 * amplitude
    sensitivities = amplitude * np.array([
        0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25
    ])
    # 2 phase encoding + velocity + amplitude + 3 state values + adaptation + 8 weights.
    vector = np.array([
        np.sin(phase), np.cos(phase), phase_velocity, amplitude,
        alertness, focus, energy, adaptation, *sensitivities,
    ], dtype=float)
    assert vector.shape == (16,)
    return vector
