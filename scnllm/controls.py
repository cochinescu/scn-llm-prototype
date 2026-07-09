"""Active-control modulation arms matching the paper's evaluation design."""

from __future__ import annotations

import numpy as np


def scn_modulation(phase: np.ndarray) -> np.ndarray:
    """Smooth behavioural proxy driven by oscillator phase."""
    return 0.5 + 0.25 * np.cos(phase - np.pi)


def random_modulation(size: int, seed: int = 23) -> np.ndarray:
    """Same broad range as SCN modulation, but unentrained and stochastic."""
    rng = np.random.default_rng(seed)
    return np.clip(0.5 + rng.normal(0.0, 0.18, size=size), 0.0, 1.0)


def heuristic_modulation(clock_phase: np.ndarray) -> np.ndarray:
    """A fixed clock-time lookup: morning/day high, night low; no re-locking."""
    local_hour = (clock_phase % (2 * np.pi)) * 24 / (2 * np.pi)
    return np.where((local_hour >= 8.0) & (local_hour < 20.0), 0.75, 0.25)
