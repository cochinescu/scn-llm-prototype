# SCN-LLM Prototype

A small, single-machine **synthetic reference implementation** of the SCN
(suprachiasmatic-nucleus) inspired temporal module for large language models: a
coupled Van der Pol oscillator "clock," phase-response entrainment to a synthetic
environmental *zeitgeber*, an analytic phase → 16-D temporal-state map, and a toy
retrieval path that the temporal state modulates. It reproduces four
mechanism-and-cost figures with one command.

This code accompanies the SCN-LLM paper as a reproducibility artifact. It is a
*simulation plus microbenchmarks* of the temporal module, nothing more.

## What this does and does not show

**It measures** — on synthetic, labelled inputs and on one machine:

- that an ensemble of coupled oscillators **synchronizes** into a coherent clock
  and free-runs near its intrinsic period;
- that the clock **entrains** (and, after a schedule shift, **re-entrains**) to an
  independent 24 h zeitgeber, driven only by the observable zeitgeber magnitude —
  the ground-truth phase is never fed into the dynamics, only used as an
  evaluation label;
- the **compute and latency** of the temporal primitives (with a clearly labelled
  system-overhead *projection*, not a measured system overhead);
- that an oscillator-driven modulation arm **re-locks** to a shifted schedule
  where a fixed clock-time lookup does not.

**It does NOT measure** believability, engagement, trust, retrieval quality,
real-sensor performance, LLM-training cost, or real RAG-system latency. The
state map is an analytic first-cut, not a trained behavioural model. These
boundaries are intentional; see the "honesty boundary" section of
[`results/RESULTS.md`](results/RESULTS.md).

## Quickstart

```sh
python -m venv .venv
.venv/bin/pip install -e ".[dev]"

# Regenerate every CSV, figure, and results summary from one fixed seed:
.venv/bin/python scripts/reproduce.py

# Run the test suite:
.venv/bin/python -m pytest -q
```

Requires Python ≥ 3.11 (NumPy, SciPy, Matplotlib). Output lands in `results/`.

## The four figures

1. **`fig1_compute_overhead.png`** — measured per-update primitive cost plus a
   labelled system-overhead projection under a stated denominator.
2. **`fig2_latency.png`** — added-module latency breakdown and toy-retrieval
   scaling (baseline vs phase-modulated).
3. **`fig3_entrainment.png`** — free-running period distribution across seeds and
   the re-entrainment transient after a schedule shift.
4. **`fig4_controls.png`** — the oscillator arm versus random and fixed-clock
   controls, discriminated by re-locking (lag shift) across the schedule shift.

## Repository layout

```
scnllm/         core modules (oscillator, zeitgeber, PRC helpers, state map,
                toy retrieval, active controls, measurement harness)
scripts/        reproduce.py (one-command pipeline) and plot.py (figures)
tests/          pytest suite for the mechanism-level properties
results/        regenerated CSVs, figures, and RESULTS.md
benchmark/      standalone, versioned phase-tracking benchmark (see below)
```

## Benchmark

[`benchmark/`](benchmark/) is a **standalone, independently-citable** phase-tracking
benchmark (its own **CC BY 4.0** license and Zenodo DOI): a frozen, versioned
synthetic zeitgeber dataset (`v1.0/`) plus a **standard-library-only** scorer, so a
temporal / clock-inference system in *any* language can be graded on the same task —
inferring the environmental phase from the observable zeitgeber alone. The oscillator
in this repo is provided as the reference baseline. See
[`benchmark/README.md`](benchmark/README.md).

## Reproducibility

Everything derives from a single master seed in `scripts/reproduce.py`.
`results/schedule_meta.json` records every generative parameter and column
definition, so the synthetic schedules can be regenerated or scored
independently. Same process + same seed is repeatable to within the ODE
solver's numerical tolerance.

## Citation

If you use this software, please cite it via [`CITATION.cff`](CITATION.cff) and
the archived release DOI (added on first Zenodo release).

## License

MIT — see [`LICENSE`](LICENSE).
