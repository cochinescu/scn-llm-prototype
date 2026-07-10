"""Language-agnostic scorer for the SCN temporal-entrainment benchmark.

A predictor -- in ANY language -- reads ``schedule.csv``, infers the
environmental phase at each timestamp, emits a JSON file of its predictions, and
this script grades it against ``labels.json``. Only the Python standard library
is used, so the scorer is trivially portable and the metric definitions are the
spec.

Predictor output format (JSON):

    {"predictions": [{"time_hours": <float>, "phase": <float radians>}, ...]}

(A bare list of the same objects is also accepted.) Every timestamp in
``labels.json`` must be covered; ``phase`` is in radians (any real value, it is
wrapped internally). Pairs are matched to labels by ``time_hours`` (6-dp).

Metrics (all circular; identical to the reference implementation's definitions):

  raw_phase_error_percent   mean |wrap(pred - truth)| / 2pi * 100  -- PRIMARY,
                            target-facing; includes the physical entrainment
                            angle, reported as-is.
  entrainment_angle_degrees circular mean of (pred - truth), post-shift -- the
                            stable phase lead/lag.
  lock_residual_percent     jitter about that stable angle / 2pi * 100 -- locking
                            *tightness*, a SEPARATE metric, NOT the phase-error
                            target.
  reentrainment_time_hours  hours after the shift until the phase difference
                            returns to within 10% of a cycle of the PRE-shift
                            entrainment angle (a non-adapting predictor never
                            re-entrains, by construction).

Usage:
    python score.py --labels v1.0/labels.json --predictions my_output.json
"""

from __future__ import annotations

import argparse
import json
import math

TWO_PI = 2.0 * math.pi


def _wrap(x: float) -> float:
    """Wrap radians to [-pi, pi)."""
    return (x + math.pi) % TWO_PI - math.pi


def _circular_mean_angle(diffs: list[float]) -> float:
    s = sum(math.sin(d) for d in diffs)
    c = sum(math.cos(d) for d in diffs)
    return math.atan2(s, c)


def _mean_abs_error_percent(diffs: list[float]) -> float:
    return sum(abs(d) for d in diffs) / len(diffs) / TWO_PI * 100.0


def _reentrainment_time_hours(times, diffs, shift_at, tol_frac=0.10):
    """Hours after the shift until the phase difference returns to within
    ``tol_frac`` of a cycle of the *pre-shift* stable entrainment angle and stays
    there. NaN if it never settles.

    The reference angle is the pre-shift entrainment relationship, NOT the
    predictor's own eventual post-shift angle: a predictor that does not adapt --
    one that keeps a stable but *unshifted* offset after the schedule moves -- must
    score as never re-entraining, not as instantly recovered. (Scoring against the
    predictor's own post-shift angle would report any predictor whose difference is
    merely stable as re-entrained, including a frozen, non-adapting one.) Mirrors
    the reference implementation."""
    post_idx = [i for i, t in enumerate(times) if t >= shift_at]
    if not post_idx:
        return float("nan")
    # Stable pre-shift entrainment angle: the second half of the pre-shift window,
    # past the initial synchronization transient.
    pre_settled = [d for t, d in zip(times, diffs) if shift_at / 2 <= t < shift_at]
    if not pre_settled:
        return float("nan")
    angle = _circular_mean_angle(pre_settled)
    deviation = [abs(_wrap(d - angle)) for d in diffs]
    threshold = tol_frac * TWO_PI
    for k in post_idx:
        if all(dev < threshold for dev in deviation[k:]):
            return times[k] - shift_at
    return float("nan")


def score(labels: dict, predictions: list) -> dict:
    shift_at = float(labels["shift_at_hours"])
    truth = {round(float(g["time_hours"]), 6): float(g["target_phase"]) for g in labels["ground_truth"]}

    # Validate the submission before scoring: a scorer that silently accepts
    # NaN, duplicate, unexpected, or malformed predictions produces NaN or
    # misleading metrics instead of rejecting an invalid entry.
    pred = {}
    for i, p in enumerate(predictions):
        if not isinstance(p, dict) or "time_hours" not in p or "phase" not in p:
            raise SystemExit(f"prediction {i} is malformed: expected an object with 'time_hours' and 'phase' (got {p!r}).")
        try:
            t = round(float(p["time_hours"]), 6)
            phase = float(p["phase"])
        except (TypeError, ValueError):
            raise SystemExit(f"prediction {i} has non-numeric 'time_hours'/'phase': {p!r}.")
        if not math.isfinite(t) or not math.isfinite(phase):
            raise SystemExit(f"prediction {i} has a non-finite 'time_hours'/'phase': {p!r}. Predictions must be finite.")
        if t in pred:
            raise SystemExit(f"duplicate prediction for time_hours={t}; each timestamp must appear exactly once.")
        if t not in truth:
            raise SystemExit(f"prediction for unexpected time_hours={t} not in the schedule; predict only the schedule's timestamps.")
        pred[t] = phase

    missing = [t for t in truth if t not in pred]
    if missing:
        raise SystemExit(
            f"submission is missing {len(missing)} of {len(truth)} timestamps "
            f"(first missing: {sorted(missing)[0]}). Every schedule row must be predicted."
        )

    # Assemble time-ordered aligned series.
    times = sorted(truth)
    diffs = [_wrap(pred[t] - truth[t]) for t in times]

    pre = [d for t, d in zip(times, diffs) if t < shift_at]
    post = [d for t, d in zip(times, diffs) if t >= shift_at]
    if not pre or not post:
        raise SystemExit("schedule has no pre- or post-shift samples; check shift_at_hours.")

    post_angle = _circular_mean_angle(post)

    def lock_residual(seg):
        a = _circular_mean_angle(seg)
        return _mean_abs_error_percent([_wrap(d - a) for d in seg])

    return {
        "task": labels.get("task"),
        "n_samples": len(times),
        "coverage": round(len(pred) / len(truth), 4),
        "pre_shift_raw_phase_error_percent": round(_mean_abs_error_percent(pre), 4),
        "post_shift_raw_phase_error_percent": round(_mean_abs_error_percent(post), 4),
        "entrainment_angle_degrees": round(math.degrees(post_angle), 4),
        "pre_shift_lock_residual_percent": round(lock_residual(pre), 4),
        "post_shift_lock_residual_percent": round(lock_residual(post), 4),
        "reentrainment_time_hours": round(_reentrainment_time_hours(times, diffs, shift_at), 4),
        "note": (
            "raw_phase_error_percent is the primary, target-facing metric and includes "
            "the physical entrainment angle; lock_residual_percent is locking tightness, "
            "a separate metric, not a substitute for the phase-error target."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--predictions", required=True)
    args = ap.parse_args()
    with open(args.labels) as fh:
        labels = json.load(fh)
    with open(args.predictions) as fh:
        sub = json.load(fh)
    predictions = sub["predictions"] if isinstance(sub, dict) else sub
    print(json.dumps(score(labels, predictions), indent=2))


if __name__ == "__main__":
    main()
