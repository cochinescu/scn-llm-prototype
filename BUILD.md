# Building & running the prototype

The implementation behind [`README.md`](README.md): a single-machine Python
simulation + microbenchmarks of the SCN temporal module. Everything is
test-driven and reproducible from a fixed seed.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"    # numpy, scipy, matplotlib, pytest
```

Requires Python ≥ 3.11.

## Run the tests

```bash
.venv/bin/python -m pytest -q
```

## Reproduce every measurement + figure (one command)

```bash
.venv/bin/python scripts/reproduce.py
```

Outputs land in `results/`: the CSVs, the four headline figures
(`fig1..fig4_*.png`), a `schedule_meta.json` sidecar recording every generative
parameter, and a generated `RESULTS.md`. Re-render the figures alone with
`.venv/bin/python scripts/plot.py`.

## Module map (`scnllm/`)

| module | role |
|---|---|
| `oscillator.py` | coupled Van der Pol network (resistive coupling), mean-field phase read-out, observable-zeitgeber entrainment, the faithful cosine-PRC phase reduction, and the entrainment/lock metrics |
| `zeitgeber.py` | synthetic labelled light/activity/interaction schedules + the independent ground-truth phase `phi*` |
| `prc.py` | circular phase-difference helpers (`wrap_phase`, `phase_error`) |
| `state_vector.py` | the analytic first-cut 16-D temporal-state map |
| `rag_modulation.py` | in-memory toy retrieval + the phase-modulated reweighting path (for timing only) |
| `controls.py` | the three modulation arms — oscillator, random, and fixed-clock heuristic |
| `bench.py` | measurement harness: latency, entrainment dynamics, analytic FLOP op-count + labelled overhead projection, active-control discrimination, and the Euler-vs-RK45 reduction diagnostic |

The one-command pipeline is `scripts/reproduce.py`; figures are drawn by
`scripts/plot.py`. The standalone, independently-citable phase-tracking benchmark
lives in `benchmark/` (frozen `v1.0/` set, standard-library-only `score.py`,
`reference_solver.py`, `build_benchmark.py`, README, CC BY 4.0 LICENSE).

## Honesty caveats (also in generated `RESULTS.md`)

- All schedules are **synthetic**; the generator parameters are recorded in
  `results/schedule_meta.json`. These figures measure oscillator dynamics, phase
  modulation, and toy-retrieval cost only — **not** believability, engagement,
  trust, or retrieval quality.
- The ground-truth phase `phi*` is **never a dynamics input** (a test enforces
  it): the module infers timing from the observable zeitgeber magnitude alone, so
  non-entrainment is a possible outcome.
- The **raw phase error** (~15.8%) is the primary, target-facing metric and
  includes the physical ~55.8° **entrainment angle**; it does **not** meet a
  `<5%` phase-error target. The **lock residual** (~3.9%) meets `<5%`, but it
  measures locking *tightness*, not distance to the environmental phase — the two
  must be reported together, never one for the other.
- Figure 1's system-overhead figure is a **labelled projection** under a stated
  denominator (an analytic FLOP op-count vs a dense forward pass), not a measured
  system overhead; it deliberately does not reproduce the architecture's ≈15%
  design target, which is attributed to sensor/RAG machinery not built here.
- **Reproducibility split.** The logical outputs — the dynamics, controls,
  overhead-projection, and free-running-period CSVs, and the phase/reduced-phase
  trajectory — are bit-for-bit reproducible from the seed (same process). The
  **latency rows are wall-clock** microbenchmarks: they are reported as a median
  with a 95% interval, wobble a few percent run-to-run, and will not match across
  machines. Treat the added-module latency *delta* as the reportable quantity,
  not the absolute milliseconds.
