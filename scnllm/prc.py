"""Circular phase-difference helpers for the synthetic simulation.

The Algorithm-1 phase reduction actually used by the model is the cosine PRC
``dphi/dt = omega + kappa*(Z-mean)*cos(phi)``, coded inline in
``oscillator.simulate`` and ``bench`` (it is derived from the full zmag model);
this module only provides the wrapping/difference primitives those callers use.
"""

from __future__ import annotations

import numpy as np


def wrap_phase(phase: np.ndarray | float) -> np.ndarray | float:
    """Wrap radians to [-pi, pi)."""
    return (np.asarray(phase) + np.pi) % (2 * np.pi) - np.pi


def phase_error(observed: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Signed circular difference in radians."""
    return wrap_phase(np.asarray(observed) - np.asarray(target))
