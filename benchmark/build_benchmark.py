"""Build the frozen, versioned SCN temporal-entrainment benchmark.

Deterministic from a fixed seed, so the artifact is reproducible and citable.
Emits, under ``benchmark/v1.0/``:

  schedule.csv   — observable zeitgeber channels (the predictor's *input*);
                   it deliberately does NOT contain the ground-truth phase
  labels.json    — the independent ground-truth environmental phase phi*(t)
                   plus task metadata (the scorer's *answer key*)
  MANIFEST.json  — schema/version, seeds, generator params, sha256 of the above

The task is *phase-tracking*: infer the environmental phase phi*(t) from the
observable zeitgeber time series alone (phi* is never an input), across a
24.2 h-intrinsic / 24.0 h-zeitgeber detuning and a mid-run schedule shift.

Run:  python benchmark/build_benchmark.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from scnllm.zeitgeber import make_schedule

OUT = HERE / "v1.0"
SCHEMA_VERSION = "scnllm-entrainment-v1"
BENCHMARK_VERSION = "1.0"
# Seeds match scripts/reproduce.py so the reference solver reproduces the
# documented baseline: MASTER_SEED for the schedule, MASTER_SEED+1 for the
# oscillator (used by reference_solver.py, NOT by the benchmark data itself).
MASTER_SEED = 20260709
OSCILLATOR_SEED = MASTER_SEED + 1
PARAMS = dict(
    duration_hours=24 * 18,
    sample_hours=1 / 6,
    shift_at_hours=24 * 9,
    shift_hours=6.0,
    seed=MASTER_SEED,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dump_json(obj, path: Path) -> None:
    with path.open("w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    s = make_schedule(**PARAMS)

    # schedule.csv -- the INPUT: observable channels only, no phi*.
    sched_path = OUT / "schedule.csv"
    with sched_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["time_hours", "light", "activity", "interaction", "value"])
        for t, li, ac, inx, v in zip(s.time_hours, s.light, s.activity, s.interaction, s.value):
            writer.writerow([f"{t:.6f}", f"{li:.10f}", f"{ac:.10f}", f"{inx:.10f}", f"{v:.10f}"])

    # labels.json -- the ANSWER KEY: independent ground-truth phase + task meta.
    labels = {
        "benchmarkVersion": BENCHMARK_VERSION,
        "schemaVersion": SCHEMA_VERSION,
        "task": "phase-tracking",
        "description": (
            "Infer the environmental phase phi*(t) from the observable zeitgeber "
            "channels in schedule.csv. phi* is defined by the imposed 24.0 h clock "
            "and is independent of any model; it is never an input. Score with "
            "score.py. Do NOT read this file inside your predictor."
        ),
        "shift_at_hours": PARAMS["shift_at_hours"],
        "shift_hours": PARAMS["shift_hours"],
        "zeitgeber_clock_period_hours": 24.0,
        "sample_hours": PARAMS["sample_hours"],
        "n_samples": int(len(s.time_hours)),
        "phase_convention": "radians in [0, 2*pi); forward clock; target_phase = 2*pi*((t - shift)%24)/24",
        "ground_truth": [
            {"time_hours": round(float(t), 6), "target_phase": round(float(p), 10)}
            for t, p in zip(s.time_hours, s.target_phase)
        ],
    }
    _dump_json(labels, OUT / "labels.json")

    manifest = {
        "schemaVersion": SCHEMA_VERSION,
        "benchmarkVersion": BENCHMARK_VERSION,
        "task": "phase-tracking",
        "seed": MASTER_SEED,
        "reference_oscillator_seed": OSCILLATOR_SEED,
        "params": PARAMS,
        "detuning_note": "intrinsic oscillator 24.2 h vs zeitgeber clock 24.0 h; a mid-run 6 h shift at hour 216",
        "n_samples": int(len(s.time_hours)),
        "sha256": {
            "schedule.csv": _sha256(sched_path),
            "labels.json": _sha256(OUT / "labels.json"),
        },
    }
    _dump_json(manifest, OUT / "MANIFEST.json")
    print(
        f"wrote benchmark v{BENCHMARK_VERSION}: {len(s.time_hours)} samples over "
        f"{PARAMS['duration_hours']:.0f} h, 6 h shift at hour {PARAMS['shift_at_hours']} -> {OUT}"
    )


if __name__ == "__main__":
    main()
