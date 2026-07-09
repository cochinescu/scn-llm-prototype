"""Small, synthetic reference implementation of the SCN temporal module."""

from .oscillator import OscillatorConfig, SimulationResult, simulate
from .zeitgeber import ZeitgeberSchedule, make_schedule

__all__ = ["OscillatorConfig", "SimulationResult", "ZeitgeberSchedule", "make_schedule", "simulate"]
