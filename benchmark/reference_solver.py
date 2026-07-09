"""Reference solver over the benchmark (worked example).

Reads ``schedule.csv``, runs THIS repo's coupled Van der Pol oscillator on the
observable zeitgeber (never reading ``labels.json``), and writes a
``predictions.json`` in the format ``score.py`` grades. It is the reference
baseline the benchmark documents; running it then scoring it demonstrates the
benchmark end-to-end and reproduces the numbers in the repo's ``RESULTS.md``.

Run:
    python benchmark/reference_solver.py --schedule v1.0/schedule.csv --out predictions.json
    python benchmark/score.py --labels v1.0/labels.json --predictions predictions.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from scnllm.oscillator import simulate
from scnllm.zeitgeber import ZeitgeberSchedule


def _load_schedule(path: str) -> ZeitgeberSchedule:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    col = lambda name: np.array([float(r[name]) for r in rows])
    time = col("time_hours")
    # target_phase is deliberately NOT in the input and is NOT used by the
    # dynamics; a zero placeholder keeps the model inferring timing from `value`.
    return ZeitgeberSchedule(
        time, col("light"), col("activity"), col("interaction"), col("value"),
        np.zeros_like(time), 0.0,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--manifest", default=str(HERE / "v1.0" / "MANIFEST.json"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.manifest) as fh:
        seed = int(json.load(fh)["reference_oscillator_seed"])
    schedule = _load_schedule(args.schedule)
    result = simulate(schedule, seed=seed)
    predictions = [
        {"time_hours": round(float(t), 6), "phase": float(p)}
        for t, p in zip(result.time_hours, result.phase)
    ]
    with open(args.out, "w") as fh:
        json.dump({"predictions": predictions}, fh, indent=2)
    print(f"wrote {len(predictions)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
