"""Turn reproduction CSVs into the four reference-implementation figures."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def make_figures(results: Path) -> None:
    latency = rows(results / "latency.csv")
    dynamics = rows(results / "dynamics.csv")[0]
    controls = rows(results / "controls.csv")
    trajectory = rows(results / "trajectory.csv")
    periods = rows(results / "free_run_periods.csv")
    projection = rows(results / "overhead_projection.csv")

    _figure_compute(results, latency, projection)
    _figure_latency(results, latency)
    _figure_dynamics(results, dynamics, trajectory, periods)
    _figure_controls(results, controls)


def _save(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _err(rows_by: dict, key: str) -> tuple[float, list[list[float]]]:
    """Median plus an asymmetric [[low_err],[high_err]] for a matplotlib yerr."""
    med, lo, hi = float(rows_by[key]["median_ms"]), float(rows_by[key]["ci_low_ms"]), float(rows_by[key]["ci_high_ms"])
    return med, [[med - lo], [hi - med]]


def _figure_compute(results: Path, latency: list[dict[str, str]], projection: list[dict[str, str]]) -> None:
    by = {row["component"]: row for row in latency if row["corpus_size"] == "0"}
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    # Left: measured primitive wall time with 95% intervals.
    names, meds, errs = [], [], [[], []]
    for comp, label in (("state_vector", "state vector"), ("scn_update", "scn update")):
        med, err = _err(by, comp)
        names.append(label); meds.append(med); errs[0].append(err[0][0]); errs[1].append(err[1][0])
    axes[0].bar(names, meds, yerr=errs, capsize=5, color=["#315b7d", "#4d8b7a"])
    axes[0].set_ylabel("Median wall time (ms / update)")
    axes[0].set_title("Measured primitive cost (95% CI)")
    # Right: projected system overhead (%) vs model size. Drawn as markers with a
    # stem (NOT bars) so it is not visually conflated with the measured quantities.
    labels = [f"{float(r['model_params_billions']):.0f}B" for r in projection]
    overhead = [float(r["projected_overhead_percent"]) for r in projection]
    xp = np.arange(len(labels))
    axes[1].vlines(xp, min(overhead) / 5, overhead, color="#a8663f", linestyle=":", linewidth=1)
    axes[1].scatter(xp, overhead, marker="D", s=70, color="#a8663f", zorder=3, label="projection (not measured)")
    axes[1].set_xticks(xp, labels)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Projected overhead (%, log)")
    axes[1].set_title("Projection under stated denominator")
    axes[1].legend(fontsize=8)
    fig.suptitle("Temporal-module compute: measured primitive + labelled projection")
    plt.figtext(0.5, 0.005, "Projection (not measured); oscillator/state primitives only — does not reproduce the paper's ~15% system budget. See RESULTS.md.", ha="center", fontsize=7)
    _save(results / "fig1_compute_overhead.png")


def _figure_latency(results: Path, latency: list[dict[str, str]]) -> None:
    by0 = {row["component"]: row for row in latency if row["corpus_size"] == "0"}
    retrieval = [row for row in latency if row["component"].startswith("retrieval")]
    sizes = sorted({int(row["corpus_size"]) for row in retrieval})
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    # Left: per-primitive breakdown with 95% CI error bars.
    prim_names, prim_meds, prim_err = [], [], [[], []]
    small = str(sizes[0])
    by_small = {(r["component"], r["corpus_size"]): r for r in retrieval}
    for comp, label, src in (
        ("state_vector", "state vector", by0.get("state_vector")),
        ("scn_update", "scn update", by0.get("scn_update")),
        ("retrieval_modulated", f"RAG reweight\n(corpus {small})", by_small.get(("retrieval_modulated", small))),
    ):
        med = float(src["median_ms"]); lo = float(src["ci_low_ms"]); hi = float(src["ci_high_ms"])
        prim_names.append(label); prim_meds.append(med); prim_err[0].append(med - lo); prim_err[1].append(hi - med)
    axes[0].bar(prim_names, prim_meds, yerr=prim_err, capsize=5, color=["#315b7d", "#4d8b7a", "#8a6d3b"])
    axes[0].set_ylabel("Median latency (ms)")
    axes[0].set_title("Added-module latency breakdown (95% CI)")
    axes[0].tick_params(axis="x", labelsize=8)
    # Right: retrieval scaling, baseline overlaid, with CI error bars.
    def series(kind):
        med, err = [], [[], []]
        for size in sizes:
            r = next(r for r in retrieval if r["component"] == kind and int(r["corpus_size"]) == size)
            m, lo, hi = float(r["median_ms"]), float(r["ci_low_ms"]), float(r["ci_high_ms"])
            med.append(m); err[0].append(m - lo); err[1].append(hi - m)
        return med, err
    base_med, base_err = series("retrieval_baseline")
    mod_med, mod_err = series("retrieval_modulated")
    x = np.arange(len(sizes)); width = 0.35
    axes[1].bar(x - width / 2, base_med, width, yerr=base_err, capsize=4, label="baseline", color="#9aabb9")
    axes[1].bar(x + width / 2, mod_med, width, yerr=mod_err, capsize=4, label="phase-modulated", color="#315b7d")
    axes[1].set_xticks(x, sizes)
    axes[1].set_xlabel("Toy corpus size")
    axes[1].set_ylabel("Median latency (ms)")
    axes[1].set_title("Retrieval scaling")
    axes[1].legend(fontsize=8)
    fig.suptitle("Added module latency (single-machine reference)")
    _save(results / "fig2_latency.png")


def _figure_dynamics(results: Path, dynamics: dict[str, str], trajectory: list[dict[str, str]], periods: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    # Left panel: free-running period distribution across seeds.
    values = [float(r["free_running_period_hours"]) for r in periods]
    axes[0].hist(values, bins=8, color="#4d8b7a", edgecolor="white")
    axes[0].axvline(24.2, color="black", linestyle="--", linewidth=1, label="intrinsic 24.2 h")
    axes[0].set_xlabel("Free-running period (hours)")
    axes[0].set_ylabel(f"Count (n={len(values)} seeds)")
    axes[0].set_title("Endogenous period distribution")
    axes[0].legend(fontsize=8)
    # Right panel: re-entrainment after the schedule shift.
    time = np.array([float(row["time_hours"]) for row in trajectory])
    phase = np.unwrap(np.array([float(row["phase"]) for row in trajectory]))
    target = np.unwrap(np.array([float(row["target_phase"]) for row in trajectory]))
    start = int(np.searchsorted(time, max(0.0, time[-1] - 24 * 11)))
    shift_hour = 216.0
    axes[1].plot(time[start:], phase[start:] / (2 * np.pi), label="SCN mean-field phase", color="#315b7d")
    axes[1].plot(time[start:], target[start:] / (2 * np.pi), label="independent schedule phase", color="#d8793f", linestyle="--")
    if time[start] <= shift_hour <= time[-1]:
        axes[1].axvline(shift_hour, color="black", linestyle=":", linewidth=1)
        ylo, yhi = axes[1].get_ylim()
        axes[1].annotate("6 h schedule shift", xy=(shift_hour, ylo), xytext=(shift_hour + 4, ylo + 0.4 * (yhi - ylo)), fontsize=8)
    axes[1].set_xlabel("Simulation time (hours)")
    axes[1].set_ylabel("Unwrapped cycles")
    axes[1].set_title(
        "Re-entrainment after 6 h shift\n"
        f"raw phase error {float(dynamics['post_shift_raw_phase_error_percent']):.1f}% "
        f"(angle {float(dynamics['entrainment_angle_degrees']):.1f}°) · "
        f"lock residual {float(dynamics['post_shift_lock_residual_percent']):.1f}% · "
        f"re-lock {float(dynamics['reentrainment_time_hours']):.0f} h",
        fontsize=9,
    )
    axes[1].legend(fontsize=8)
    _save(results / "fig3_entrainment.png")


def _figure_controls(results: Path, controls: list[dict[str, str]]) -> None:
    arms = [row["arm"] for row in controls]
    pre = [float(row["pre_shift_peak_correlation"]) for row in controls]
    post = [float(row["post_shift_peak_correlation"]) for row in controls]
    lag_shift = [float(row["lag_shift_hours"]) for row in controls]
    x = np.arange(len(arms))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    # Left: peak cross-correlation with the zeitgeber (structured-ness).
    width = 0.38
    axes[0].bar(x - width / 2, pre, width, label="pre-shift", color="#9aabb9")
    axes[0].bar(x + width / 2, post, width, label="post-shift", color="#315b7d")
    axes[0].set_xticks(x, arms)
    axes[0].set_ylabel("Peak |cross-correlation|")
    axes[0].set_title("Structured like the zeitgeber?")
    axes[0].legend(fontsize=8)
    # Right: lag shift across the schedule shift (the re-locking discriminator).
    axes[1].bar(x, lag_shift, color=["#315b7d", "#a8b7c3", "#d8793f"])
    axes[1].axhline(6, color="black", linestyle="--", linewidth=1, label="6 h schedule shift")
    axes[1].set_xticks(x, arms)
    axes[1].set_ylabel("Zeitgeber lag shift (hours)")
    axes[1].set_title("Re-locks after shift? (SCN ~0, heuristic ~6 h)")
    axes[1].legend(fontsize=8)
    fig.suptitle("SCN modulation versus active controls")
    _save(results / "fig4_controls.png")


if __name__ == "__main__":
    make_figures(ROOT / "results")
