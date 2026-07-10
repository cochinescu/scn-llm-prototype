# SCN Temporal-Entrainment Benchmark

A frozen, versioned, seed-reproducible **phase-tracking** task: infer the
environmental phase `phi*(t)` of a synthetic 24.0 h zeitgeber from its
**observable channels alone**, across a 24.2 h-intrinsic / 24.0 h-zeitgeber
detuning and a mid-run 6 h schedule shift. It benchmarks *any* temporal /
clock-inference system against the same synthetic circadian task, independently
of the reference implementation in this repository. It accompanies the SCN-LLM
paper but is released as a standalone, citable artifact: a predictor written in
**any language** can be scored against it without touching this repo's Python.

## Contents (`v1.0/`)

| file | role |
|---|---|
| `schedule.csv` | the predictor's **input** — `time_hours` + observable zeitgeber channels (`light`, `activity`, `interaction`, `value`). **No ground-truth phase.** |
| `labels.json` | the **answer key** — the independent ground-truth phase `phi*(t)` + task metadata (do **not** read this in your predictor) |
| `MANIFEST.json` | schema/version, seeds, generator params, and SHA-256 of the two files above |

`v1.0` is **2593 samples over 432 h** (10-minute spacing), with a **6 h schedule
shift at hour 216**. It is deterministic from `seed = 20260709`.

## Format

**Input row** (`schedule.csv`):

```
time_hours,light,activity,interaction,value
0.000000,0.0053344792,0.0795595087,0.4111122905,0.0884686598
```

`value = 0.55*light + 0.30*activity + 0.15*interaction` is the aggregate
zeitgeber magnitude `Z(t)`. `phi*` is **not** provided — the task is to infer
timing from the temporal structure of these observable channels.

**Label** (`labels.json` → `ground_truth[]`):

```json
{ "time_hours": 0.0, "target_phase": 0.0 }
```

`target_phase` is `2*pi*((t - shift) % 24) / 24`, radians in `[0, 2*pi)`, defined
by the imposed clock and independent of any model. `labels.json` also carries
`shift_at_hours`, `shift_hours`, and `sample_hours`.

## Scoring your predictor

Your predictor reads `schedule.csv` and writes its inferred phase per timestamp:

```json
{ "predictions": [ { "time_hours": 0.0, "phase": 1.234 }, ... ] }
```

`phase` is in radians (any real value; it is wrapped internally). Every schedule
timestamp must be predicted. Then:

```bash
python score.py --labels v1.0/labels.json --predictions your_output.json
```

`score.py` is **standard-library-only** — it needs nothing but Python, and it is
the only script that runs against the benchmark files alone. The two scripts
below (`reference_solver.py`, `build_benchmark.py`) are *conveniences* that
reproduce and regenerate the dataset; they `import scnllm` and therefore require
the separate **software release** (`pip install` the reference implementation,
concept DOI [10.5281/zenodo.21286077](https://doi.org/10.5281/zenodo.21286077)),
which is **not** bundled in this standalone benchmark archive. Scoring your own
predictor never needs them.

`score.py`'s metric definitions are the spec:

| metric | meaning |
|---|---|
| `raw_phase_error_percent` (pre/post) | **primary, target-facing**: mean `\|wrap(pred − truth)\|` as % of a cycle. Includes the physical entrainment angle, reported as-is. |
| `entrainment_angle_degrees` | circular mean of `(pred − truth)`, post-shift — the stable phase lead/lag. |
| `lock_residual_percent` (pre/post) | jitter about that stable angle — locking **tightness**, a *separate* metric, **not** a substitute for the phase-error target. |
| `reentrainment_time_hours` | hours after the shift until the phase difference settles within 10% of a cycle of its stable angle. |

## Worked example (reference baseline)

Using this repo's coupled-oscillator model as the reference predictor
(`reference_solver.py` needs the `scnllm` software release installed — see the
note above; `score.py` does not):

```bash
python reference_solver.py --schedule v1.0/schedule.csv --out predictions.json
python score.py --labels v1.0/labels.json --predictions predictions.json
```

reproduces the numbers documented in the repo's `results/RESULTS.md`:

```
post_shift_raw_phase_error_percent : 15.76   (the physical 55.8 deg entrainment angle)
entrainment_angle_degrees          : 55.80
pre/post_shift_lock_residual_percent : 1.96 / 3.86
reentrainment_time_hours           : 25.67
```

**Read these two numbers together.** The raw phase error (~15.8%) does **not**
meet a `<5%` phase-error target — it is dominated by the constant, physically
plausible ~55.8° entrainment angle. The lock residual (~3.9%) meets `<5%`, but it
measures *locking tightness*, not distance to the environmental phase. A result
that quotes only the lock residual as meeting a `<5%` phase-error target is
misreporting the task.

## Reproducing / extending

```bash
python build_benchmark.py    # regenerates v1.0/ deterministically from seed 20260709
```

`build_benchmark.py` also `import`s `scnllm` (it drives the same generator as the
software release), so it too requires that package installed; it is not needed to
*use* the frozen `v1.0/` data, only to regenerate it. The SHA-256 digests in
`MANIFEST.json` pin the released files. To propose a new
version, bump the params/seed and `benchmarkVersion`, and keep old versions
frozen.

## Scope & honesty

- The schedules are **synthetic**; all generator parameters are in
  `MANIFEST.json`. This measures phase-tracking of a synthetic zeitgeber only —
  **not** believability, engagement, trust, or retrieval quality.
- `v1.0` perturbs only the zeitgeber **magnitude**; a timing-irregular regime
  (shift-work-style phase disruption) is future work, kept out so no result
  overstates robustness.
- `phi*` is never an input, so non-entrainment is a possible outcome — a
  predictor that ignores the zeitgeber will score poorly, by construction.

## License

The data and code in this directory are released under **CC BY 4.0**
(`LICENSE`). Attribution: cite the accompanying paper and this benchmark's DOI.

## Citation

> S. Cochinescu. *SCN Temporal-Entrainment Benchmark v1.0.* Zenodo, 2026.
> DOI: [10.5281/zenodo.21286179](https://doi.org/10.5281/zenodo.21286179)
> (concept DOI). Accompanies the SCN-LLM paper; the reference implementation is
> archived at [10.5281/zenodo.21286077](https://doi.org/10.5281/zenodo.21286077).

The benchmark has its **own** Zenodo DOI, separate from the code repository's
concept DOI, so it is independently citable.
