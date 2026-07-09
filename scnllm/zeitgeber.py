"""Versioned synthetic zeitgeber schedules and independent phase labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ZeitgeberSchedule:
    """Synthetic inputs plus an analytic environmental phase label.

    ``target_phase`` is intentionally derived from the imposed 24-hour schedule,
    never from the oscillator being evaluated.
    """

    time_hours: np.ndarray
    light: np.ndarray
    activity: np.ndarray
    interaction: np.ndarray
    value: np.ndarray
    target_phase: np.ndarray
    shift_hour: float
    schema_version: str = "scnllm-zeitgeber-v1"


def _daily_profile(clock_hour: np.ndarray, centre: float, width: float) -> np.ndarray:
    distance = np.minimum(np.abs(clock_hour - centre), 24 - np.abs(clock_hour - centre))
    return np.exp(-0.5 * (distance / width) ** 2)


def make_schedule(
    duration_hours: float = 24 * 18,
    sample_hours: float = 1 / 6,
    shift_at_hours: float | None = 24 * 9,
    shift_hours: float = 6.0,
    irregular: bool = False,
    seed: int = 7,
) -> ZeitgeberSchedule:
    """Create labelled light/activity/interaction signals on a 24 h clock.

    The schedule moves by ``shift_hours`` after ``shift_at_hours``.  A small,
    seeded perturbation is optional to represent irregular schedules.
    """
    rng = np.random.default_rng(seed)
    time = np.arange(0.0, duration_hours + sample_hours / 2, sample_hours)
    shift = np.zeros_like(time) if shift_at_hours is None else np.where(time >= shift_at_hours, shift_hours, 0.0)
    clock = (time - shift) % 24.0
    light = _daily_profile(clock, centre=13.0, width=3.4)
    activity = _daily_profile(clock, centre=15.0, width=4.0)
    interaction = _daily_profile(clock, centre=20.0, width=3.0)
    if irregular:
        noise = rng.normal(0.0, 0.10, size=time.size)
        light = np.clip(light + noise, 0.0, 1.0)
        activity = np.clip(activity + 0.8 * noise, 0.0, 1.0)
        interaction = np.clip(interaction + 0.6 * noise, 0.0, 1.0)
    value = 0.55 * light + 0.30 * activity + 0.15 * interaction
    # Environmental schedule phase; it is independent of model parameters.
    target_phase = (2 * np.pi * clock / 24.0) % (2 * np.pi)
    return ZeitgeberSchedule(time, light, activity, interaction, value, target_phase, shift_hours)
