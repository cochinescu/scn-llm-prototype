"""Tests for the standalone benchmark scorer (``benchmark/score.py``).

The scorer is stdlib-only and lives outside the ``scnllm`` package, so it is
loaded by path here. These cover the smoke path (a valid submission scores) and
the input-validation guards (NaN, duplicate, unexpected, malformed, and missing
predictions are rejected rather than silently producing NaN or wrong metrics).
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

_SCORE_PATH = Path(__file__).resolve().parent.parent / "benchmark" / "score.py"
_spec = importlib.util.spec_from_file_location("benchmark_score", _SCORE_PATH)
score_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score_mod)


def _labels():
    # Minimal schedule: 4 timestamps spanning a shift at hour 2.
    return {
        "task": "phase-tracking",
        "shift_at_hours": 2.0,
        "ground_truth": [
            {"time_hours": 0.0, "target_phase": 0.0},
            {"time_hours": 1.0, "target_phase": 0.2},
            {"time_hours": 2.0, "target_phase": 0.4},
            {"time_hours": 3.0, "target_phase": 0.6},
        ],
    }


def _valid_predictions():
    return [
        {"time_hours": 0.0, "phase": 0.0},
        {"time_hours": 1.0, "phase": 0.2},
        {"time_hours": 2.0, "phase": 0.4},
        {"time_hours": 3.0, "phase": 0.6},
    ]


def test_valid_submission_scores():
    out = score_mod.score(_labels(), _valid_predictions())
    assert out["coverage"] == 1.0
    assert out["n_samples"] == 4
    assert math.isclose(out["post_shift_raw_phase_error_percent"], 0.0, abs_tol=1e-9)
    assert "reentrainment_time_hours" in out


def test_rejects_nan_phase():
    preds = _valid_predictions()
    preds[1]["phase"] = float("nan")
    with pytest.raises(SystemExit):
        score_mod.score(_labels(), preds)


def test_rejects_duplicate_timestamp():
    preds = _valid_predictions()
    preds.append({"time_hours": 1.0, "phase": 0.3})
    with pytest.raises(SystemExit):
        score_mod.score(_labels(), preds)


def test_rejects_unexpected_timestamp():
    preds = _valid_predictions()
    preds.append({"time_hours": 99.0, "phase": 0.0})
    with pytest.raises(SystemExit):
        score_mod.score(_labels(), preds)


def test_rejects_malformed_object():
    preds = _valid_predictions()
    preds[0] = {"time_hours": 0.0}  # missing "phase"
    with pytest.raises(SystemExit):
        score_mod.score(_labels(), preds)


def test_rejects_missing_timestamps():
    preds = _valid_predictions()[:-1]  # drop the last required timestamp
    with pytest.raises(SystemExit):
        score_mod.score(_labels(), preds)
