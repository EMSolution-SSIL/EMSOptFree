"""
run_analysis.py
The MIT License (MIT)
Copyright © 2026 Science Solutions International Laboratory, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from itertools import pairwise
from logging import getLogger
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
import yaml

from utils.individuals_io import load_individuals_csv, resolve_outcome_filepath

mpl.use("Agg")
import matplotlib.pyplot as plt

logger = getLogger(__name__)
TWO_OBJECTIVES = 2

# please list optimizers' decorator names for analyze_runs
SO_OPTIMIZER_NAMES = {"cmaes", "ga", "gradient_update"}
MO_OPTIMIZER_NAMES = {"nsga2", "sa_nsga2", "decomposition_ensemble", "moead"}

RUN_INFO_FILENAME = "run_info.yaml"
ANALYSIS_DIRNAME = "analysis"
RUN_STATUS_FILENAME = "run_status.csv"
RUN_HEALTH_FILENAME = "run_health.yaml"
SO_COMPARISON_FILENAME = "single_objective_run_comparison.csv"
SO_PLOT_FILENAME = "single_objective_runs.png"
SO_STATS_FILENAME = "single_objective_stats.yaml"
MO_HYPERVOLUME_FILENAME = "multi_objective_hypervolume.csv"
MO_HYPERVOLUME_PLOT_FILENAME = "multi_objective_hypervolume.png"
MO_PARETO_PLOT_FILENAME = "multi_objective_pareto.png"
MO_STATS_FILENAME = "multi_objective_stats.yaml"


def build_multi_run_visualizer_data(
    project_summary_dir: str | Path,
    *,
    study_name: str,
    runs_dirname: str = "runs",
    resultant_dirname: str = "resultant",
    run_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Build Graph/Table inputs for the analyze_runs GUI mode."""
    study_summary_dir = Path(project_summary_dir) / "optimization_studies" / study_name
    runs_dir = study_summary_dir / runs_dirname
    configured_mode = _infer_study_mode(project_summary_dir, study_name)
    run_infos = _collect_run_infos(runs_dir, run_ids=run_ids)
    completed_runs = [info for info in run_infos if str(info.get("status")) == "completed"]
    if not completed_runs:
        msg = f"No completed runs found for study '{study_name}'."
        raise ValueError(msg)

    objective_counts = Counter()
    per_run_individuals: dict[str, list[Any]] = {}
    for info in completed_runs:
        run_id = str(info["run_id"])
        individuals = _load_run_individuals(Path(info["run_summary_dir"]), resultant_dirname=resultant_dirname)
        if not individuals:
            continue
        per_run_individuals[run_id] = individuals
        objective_counts[len(individuals[0].metrics.objectives)] += 1

    if not objective_counts:
        msg = f"No GUI-displayable run artifacts found for study '{study_name}'."
        raise ValueError(msg)

    if configured_mode == "SO":
        filtered_individuals = []
        for individuals in per_run_individuals.values():
            filtered_individuals.extend(individuals)
        return {
            "mode": "SO",
            "num_obj": 1,
            "individuals": filtered_individuals,
        }

    common_objective_count = objective_counts.most_common(1)[0][0]
    filtered_individuals = []
    for individuals in per_run_individuals.values():
        if len(individuals[0].metrics.objectives) != common_objective_count:
            continue
        filtered_individuals.extend(individuals)
    if not filtered_individuals:
        msg = f"No compatible runs found for study '{study_name}'."
        raise ValueError(msg)
    return {
        "mode": "SO" if common_objective_count == 1 else "MO",
        "num_obj": common_objective_count,
        "individuals": filtered_individuals,
    }


def write_run_info(
    run_summary_dir: str | Path,
    *,
    run_id: str,
    study_name: str,
    status: str,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
    failure_reason: str | None = None,
) -> Path:
    """Persist run-level metadata for progress and health reporting."""
    run_dir = Path(run_summary_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "study_name": study_name,
        "status": status,
        "started_at": _serialize_datetime(started_at),
        "finished_at": _serialize_datetime(finished_at),
        "failure_reason": failure_reason,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    path = run_dir / RUN_INFO_FILENAME
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return path


def analyze_study_runs(
    project_summary_dir: str | Path,
    *,
    study_name: str,
    runs_dirname: str = "runs",
    resultant_dirname: str = "resultant",
    reference_point: list[float] | None = None,
    run_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Build study-level comparison artifacts and run statistics.

    The study summary already stores each run under `runs/<run_id>/`.
    This function reads those run-local artifacts and materializes
    study-wide comparison outputs under `analysis/`.
    """
    study_summary_dir = Path(project_summary_dir) / "optimization_studies" / study_name
    runs_dir = study_summary_dir / runs_dirname
    analysis_dir = study_summary_dir / ANALYSIS_DIRNAME
    analysis_dir.mkdir(parents=True, exist_ok=True)

    run_infos = _collect_run_infos(runs_dir, run_ids=run_ids)
    status_df = pd.DataFrame(run_infos)
    if not status_df.empty:
        status_df.to_csv(analysis_dir / RUN_STATUS_FILENAME, index=False)

    health_summary = _build_health_summary(run_infos)
    with (analysis_dir / RUN_HEALTH_FILENAME).open("w", encoding="utf-8") as f:
        yaml.safe_dump(health_summary, f, sort_keys=False)

    completed_runs = [info for info in run_infos if str(info.get("status")) == "completed"]
    result = {
        "study_name": study_name,
        "analysis_dir": str(analysis_dir),
        "health": health_summary,
        "single_objective": None,
        "multi_objective": None,
    }
    if not completed_runs:
        return result

    objective_counts = Counter()
    run_tables: dict[str, pd.DataFrame] = {}
    for info in completed_runs:
        run_dir = Path(info["run_summary_dir"])
        df = _load_run_objective_table(
            run_dir,
            resultant_dirname=resultant_dirname,
        )
        if df is None:
            continue
        run_tables[str(info["run_id"])] = df
        objective_cols = _get_objective_columns(df)
        if objective_cols:
            objective_counts[len(objective_cols)] += 1

    if not objective_counts:
        return result

    configured_mode = _infer_study_mode(project_summary_dir, study_name)

    # Select the dominant objective contract first and exclude incompatible runs
    # later so mixed studies can still produce usable aggregate outputs.
    common_objective_count = objective_counts.most_common(1)[0][0]
    if configured_mode == "SO" or (configured_mode is None and common_objective_count == 1):
        result["single_objective"] = _analyze_single_objective_runs(
            analysis_dir=analysis_dir,
            completed_runs=completed_runs,
            run_tables=run_tables,
        )
        return result

    result["multi_objective"] = _analyze_multi_objective_runs(
        analysis_dir=analysis_dir,
        completed_runs=completed_runs,
        common_objective_count=common_objective_count,
        resultant_dirname=resultant_dirname,
        reference_point=reference_point,
    )
    return result


def _analyze_single_objective_runs(
    *,
    analysis_dir: Path,
    completed_runs: list[dict[str, Any]],
    run_tables: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Summarize completed single-objective runs using best-so-far traces."""
    comparison_rows = []
    included_run_ids = []
    exclusion_reasons: dict[str, str] = {}
    best_values = []

    for info in completed_runs:
        run_id = str(info["run_id"])
        df = run_tables.get(run_id)
        if df is None:
            exclusion_reasons[run_id] = "run_records_missing"
            continue
        series = _get_single_objective_series(df)
        if series.empty:
            exclusion_reasons[run_id] = "fitness_or_objective_values_missing"
            continue
        included_run_ids.append(run_id)
        # The acceptance criteria ask for "best objective observed in the run",
        # so we keep the running minimum and also export the full trace for plotting.
        running_best = series.cummin().tolist()
        best_values.append(float(min(running_best)))
        for index, value in enumerate(running_best, start=1):
            comparison_rows.append(
                {
                    "run_id": run_id,
                    "evaluation_index": index,
                    "best_objective": float(value),
                }
            )

    stats = {
        "num_target_runs": len(completed_runs),
        "num_included_runs": len(included_run_ids),
        "num_excluded_runs": len(exclusion_reasons),
        "excluded_runs": exclusion_reasons,
        "best_objective_mean": float(np.mean(best_values)) if best_values else None,
        "best_objective_std": float(np.std(best_values, ddof=0)) if best_values else None,
    }
    comparison_df = pd.DataFrame(comparison_rows)
    if not comparison_df.empty:
        comparison_df.to_csv(analysis_dir / SO_COMPARISON_FILENAME, index=False)
        _plot_single_objective_comparison(comparison_df, analysis_dir / SO_PLOT_FILENAME)
    with (analysis_dir / SO_STATS_FILENAME).open("w", encoding="utf-8") as f:
        yaml.safe_dump(stats, f, sort_keys=False)
    return stats


def _analyze_multi_objective_runs(
    *,
    analysis_dir: Path,
    completed_runs: list[dict[str, Any]],
    common_objective_count: int,
    resultant_dirname: str,
    reference_point: list[float] | None,
) -> dict[str, Any]:
    """Collect comparable Pareto sets first, then compute quality summaries."""
    included_points, exclusion_reasons = _collect_multi_objective_points(
        completed_runs=completed_runs,
        common_objective_count=common_objective_count,
        resultant_dirname=resultant_dirname,
    )
    return _summarize_multi_objective_runs(
        analysis_dir=analysis_dir,
        completed_runs=completed_runs,
        included_points=included_points,
        exclusion_reasons=exclusion_reasons,
        common_objective_count=common_objective_count,
        reference_point=reference_point,
    )


def _collect_multi_objective_points(
    *,
    completed_runs: list[dict[str, Any]],
    common_objective_count: int,
    resultant_dirname: str,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Load one comparable non-dominated point set per completed run."""
    included_points: dict[str, np.ndarray] = {}
    exclusion_reasons: dict[str, str] = {}
    for info in completed_runs:
        run_id = str(info["run_id"])
        run_dir = Path(info["run_summary_dir"])
        points_df = _load_multi_objective_points(
            run_dir,
            resultant_dirname=resultant_dirname,
        )
        if points_df is None:
            exclusion_reasons[run_id] = "objective_table_missing"
            continue
        objective_cols = _get_objective_columns(points_df)
        if len(objective_cols) != common_objective_count:
            exclusion_reasons[run_id] = "incompatible_objective_count"
            continue
        objective_df = points_df[objective_cols].apply(pd.to_numeric, errors="coerce").dropna()
        if objective_df.empty:
            exclusion_reasons[run_id] = "objective_values_missing"
            continue
        points = objective_df.to_numpy(dtype=float)
        # Hypervolume should be evaluated on the final non-dominated set only.
        nondominated_points = _filter_nondominated(points)
        if len(nondominated_points) == 0:
            exclusion_reasons[run_id] = "pareto_front_empty"
            continue
        included_points[run_id] = nondominated_points
    return included_points, exclusion_reasons


def _summarize_multi_objective_runs(
    *,
    analysis_dir: Path,
    completed_runs: list[dict[str, Any]],
    included_points: dict[str, np.ndarray],
    exclusion_reasons: dict[str, str],
    common_objective_count: int,
    reference_point: list[float] | None,
) -> dict[str, Any]:
    """Compute hypervolume-based study statistics from comparable Pareto sets."""
    resolved_reference_point = reference_point
    if included_points and resolved_reference_point is None:
        # Fall back to a conservative reference point slightly outside the union
        # of observed fronts so a caller can omit it for quick comparisons.
        resolved_reference_point = _infer_reference_point(list(included_points.values()))

    if included_points and common_objective_count == TWO_OBJECTIVES:
        _plot_multi_objective_pareto(included_points, analysis_dir / MO_PARETO_PLOT_FILENAME)

    hv_rows = []
    if resolved_reference_point is not None:
        ref_array = np.asarray(resolved_reference_point, dtype=float)
        for run_id, points in list(included_points.items()):
            if len(ref_array) != points.shape[1]:
                exclusion_reasons[run_id] = "reference_point_dimension_mismatch"
                del included_points[run_id]
                continue
            hv = _compute_hypervolume(points, ref_array)
            if hv is None:
                exclusion_reasons[run_id] = "reference_point_not_dominated"
                del included_points[run_id]
                continue
            hv_rows.append({"run_id": run_id, "hypervolume": float(hv)})

    hv_df = pd.DataFrame(hv_rows)
    if not hv_df.empty:
        hv_df.to_csv(analysis_dir / MO_HYPERVOLUME_FILENAME, index=False)
        _plot_hypervolume(hv_df, analysis_dir / MO_HYPERVOLUME_PLOT_FILENAME)

    stats = {
        "num_target_runs": len(completed_runs),
        "num_included_runs": len(included_points),
        "num_excluded_runs": len(exclusion_reasons),
        "excluded_runs": exclusion_reasons,
        "reference_point": list(map(float, resolved_reference_point)) if resolved_reference_point is not None else None,
        "hypervolume_mean": float(hv_df["hypervolume"].mean()) if not hv_df.empty else None,
        "hypervolume_std": float(hv_df["hypervolume"].std(ddof=0)) if not hv_df.empty else None,
    }
    with (analysis_dir / MO_STATS_FILENAME).open("w", encoding="utf-8") as f:
        yaml.safe_dump(stats, f, sort_keys=False)
    return stats


def _collect_run_infos(
    runs_dir: Path,
    *,
    run_ids: list[str] | None,
) -> list[dict[str, Any]]:
    """Load lightweight per-run metadata used for status and health reporting."""
    if not runs_dir.exists():
        return []
    selected_run_ids = set(run_ids) if run_ids is not None else None
    run_infos = []
    for run_dir in sorted([child for child in runs_dir.iterdir() if child.is_dir()]):
        if selected_run_ids is not None and run_dir.name not in selected_run_ids:
            continue
        info_path = run_dir / RUN_INFO_FILENAME
        if info_path.exists():
            with info_path.open(encoding="utf-8") as f:
                payload = yaml.safe_load(f) or {}
        else:
            # Older runs may exist before `run_info.yaml` was introduced.
            payload = {"run_id": run_dir.name, "status": "unknown", "failure_reason": None}
        summary_path = run_dir / "summary.yaml"
        if summary_path.exists():
            with summary_path.open(encoding="utf-8") as f:
                summary = yaml.safe_load(f) or {}
        else:
            summary = {}
        run_infos.append(
            {
                "run_id": str(payload.get("run_id", run_dir.name)),
                "status": str(payload.get("status", "unknown")),
                "started_at": payload.get("started_at"),
                "finished_at": payload.get("finished_at"),
                "failure_reason": payload.get("failure_reason"),
                "num_iteration": summary.get("num_iteration"),
                "num_candidates_per_iteration": summary.get("num_candidates_per_iteration"),
                "num_parallel_process": summary.get("num_parallel_process"),
                "total_computation_time": summary.get("total_computation_time"),
                "run_summary_dir": str(run_dir),
            }
        )
    return run_infos


def _load_run_individuals(run_dir: Path, *, resultant_dirname: str) -> list[Any]:
    """Load run-local individuals for GUI display and attach analysis metadata."""
    table = _load_run_objective_table(run_dir, resultant_dirname=resultant_dirname)
    if table is None:
        return []
    csv_path = _resolve_run_objective_csv(run_dir, resultant_dirname=resultant_dirname)
    if csv_path is None:
        return []
    individuals = load_individuals_csv(str(csv_path))
    run_id = run_dir.name
    for index, individual in enumerate(individuals, start=1):
        resolved = resolve_outcome_filepath(individual.outcome_filepath, summary_dir=run_dir)
        individual.outcome_filepath = str(resolved) if resolved is not None else None
        individual.analysis_run_id = run_id
        individual.analysis_x_value = index
    return individuals


def _build_health_summary(run_infos: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate success/failure ratios and major failure reasons for a study."""
    total_runs = len(run_infos)
    status_counts = Counter(str(info.get("status", "unknown")) for info in run_infos)
    failed_runs = sum(1 for info in run_infos if str(info.get("status")) == "failed")
    completed_runs = sum(1 for info in run_infos if str(info.get("status")) == "completed")
    failure_reason_counts = Counter(
        str(info["failure_reason"]) for info in run_infos if info.get("failure_reason") not in (None, "")
    )
    return {
        "num_runs": total_runs,
        "status_counts": dict(status_counts),
        "success_rate": 0.0 if total_runs == 0 else completed_runs / total_runs,
        "failure_rate": 0.0 if total_runs == 0 else failed_runs / total_runs,
        "failure_reason_counts": dict(failure_reason_counts),
    }


def _load_multi_objective_points(
    run_dir: Path,
    *,
    resultant_dirname: str,
) -> pd.DataFrame | None:
    """Resolve the best available objective table for a multi-objective run.

    We rely on run-local `resultant/` artifacts only so the analysis layer stays
    independent from the optional heavy `records/` exports.
    """
    resultant_dir = run_dir / resultant_dirname
    csv_candidates = sorted(path for path in resultant_dir.glob("*.csv") if path.name != "surrogate_individuals.csv")
    for candidate in csv_candidates:
        df = pd.read_csv(candidate)
        if _get_objective_columns(df):
            return df
    return None


def _load_run_objective_table(
    run_dir: Path,
    *,
    resultant_dirname: str,
) -> pd.DataFrame | None:
    """Load a representative objective table from run-local resultant artifacts."""
    multi_df = _load_multi_objective_points(run_dir, resultant_dirname=resultant_dirname)
    if multi_df is not None:
        return multi_df
    return _load_single_objective_table(run_dir, resultant_dirname=resultant_dirname)


def _resolve_run_objective_csv(run_dir: Path, *, resultant_dirname: str) -> Path | None:
    """Resolve the concrete CSV file used as the GUI source for a run."""
    resultant_dir = run_dir / resultant_dirname
    best_csv = run_dir / resultant_dirname / "best_individual.csv"
    if best_csv.exists():
        df = pd.read_csv(best_csv)
        if _get_objective_columns(df):
            return best_csv
    csv_candidates = sorted(path for path in resultant_dir.glob("*.csv") if path.name != "surrogate_individuals.csv")
    for candidate in csv_candidates:
        df = pd.read_csv(candidate)
        if _get_objective_columns(df):
            return candidate
    return None


def _get_single_objective_series(df: pd.DataFrame) -> pd.Series:
    """Return the scalar SO trace, preferring persisted fitness when available."""
    if "fitness" in df.columns:
        fitness = pd.to_numeric(df["fitness"], errors="coerce").dropna()
        if not fitness.empty:
            return fitness
    objective_cols = _get_objective_columns(df)
    if not objective_cols:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[objective_cols[0]], errors="coerce").dropna()


def _infer_study_mode(project_summary_dir: str | Path, study_name: str) -> str | None:
    """Infer SO/MO from the sibling study optimization.yaml when available."""
    study_dir = Path(project_summary_dir).parent / "optimization_studies" / study_name
    optimization_yaml = study_dir / "optimization.yaml"
    if not optimization_yaml.exists():
        return None
    with optimization_yaml.open(encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    optimizer_name = str((config.get("optimizer") or {}).get("name") or "").strip()
    if optimizer_name in SO_OPTIMIZER_NAMES:
        return "SO"
    if optimizer_name in MO_OPTIMIZER_NAMES:
        return "MO"
    return None


def _load_single_objective_table(
    run_dir: Path,
    *,
    resultant_dirname: str,
) -> pd.DataFrame | None:
    """Resolve the best available history table for a single-objective run.

    Single-objective runs are analyzed from the run-local `best_individual.csv`
    trace exported under `resultant/`.
    """
    best_csv = run_dir / resultant_dirname / "best_individual.csv"
    if best_csv.exists():
        df = pd.read_csv(best_csv)
        if _get_objective_columns(df):
            return df
    return None


def _plot_single_objective_comparison(df: pd.DataFrame, dst_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for run_id, run_df in df.groupby("run_id"):
        ax.plot(run_df["evaluation_index"], run_df["best_objective"], label=str(run_id))
    ax.set_xlabel("Evaluation Index")
    ax.set_ylabel("Best Objective")
    ax.set_title("Single Objective Multi-Run Comparison")
    ax.grid(visible=True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(dst_path)
    plt.close(fig)


def _plot_hypervolume(df: pd.DataFrame, dst_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df["run_id"], df["hypervolume"])
    ax.set_xlabel("Run ID")
    ax.set_ylabel("Hypervolume")
    ax.set_title("Multi-Objective Hypervolume by Run")
    ax.grid(visible=True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(dst_path)
    plt.close(fig)


def _plot_multi_objective_pareto(points_by_run: dict[str, np.ndarray], dst_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for run_id, points in points_by_run.items():
        ax.scatter(points[:, 0], points[:, 1], label=run_id)
    ax.set_xlabel("Objective 1")
    ax.set_ylabel("Objective 2")
    ax.set_title("Final Pareto Comparison")
    ax.grid(visible=True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(dst_path)
    plt.close(fig)


def _get_objective_columns(df: pd.DataFrame) -> list[str]:
    """Return objective columns in stable numeric order."""
    return sorted(col for col in df.columns if col.startswith("objective_"))


def _filter_nondominated(points: np.ndarray) -> np.ndarray:
    """Filter a minimization point set down to its non-dominated members."""
    if len(points) == 0:
        return points
    keep = np.ones(len(points), dtype=bool)
    for i, point in enumerate(points):
        if not keep[i]:
            continue
        dominates = np.all(points <= point, axis=1) & np.any(points < point, axis=1)
        if np.any(dominates):
            keep[i] = False
    return points[keep]


def _infer_reference_point(point_sets: list[np.ndarray]) -> list[float]:
    """Infer a reference point just outside the observed point cloud."""
    stacked = np.vstack(point_sets)
    max_values = np.max(stacked, axis=0)
    min_values = np.min(stacked, axis=0)
    span = np.maximum(max_values - min_values, 1.0)
    return (max_values + 0.1 * span).astype(float).tolist()


def _compute_hypervolume(points: np.ndarray, reference_point: np.ndarray) -> float | None:
    """Compute minimization hypervolume using an axis-aligned box union."""
    valid_points = [
        point.tolist() for point in points if np.all(np.isfinite(point)) and np.all(point <= reference_point)
    ]
    if not valid_points:
        return None
    boxes = [(point, reference_point.tolist()) for point in valid_points]
    return float(_union_box_volume(boxes))


def _union_box_volume(boxes: list[tuple[list[float], list[float]]]) -> float:
    """Recursively compute the union volume of axis-aligned boxes."""
    if not boxes:
        return 0.0
    dim = len(boxes[0][0])
    if dim == 1:
        intervals = sorted((box[0][0], box[1][0]) for box in boxes if box[0][0] < box[1][0])
        return _one_dim_union_length(intervals)

    split_points = sorted({bound for start, end in boxes for bound in (start[0], end[0])})
    total = 0.0
    for left, right in pairwise(split_points):
        if right <= left:
            continue
        active_boxes = []
        for start, end in boxes:
            if start[0] <= left and end[0] >= right:
                active_boxes.append((start[1:], end[1:]))
        if active_boxes:
            total += (right - left) * _union_box_volume(active_boxes)
    return total


def _one_dim_union_length(intervals: list[tuple[float, float]]) -> float:
    """Compute the covered length of 1D closed-open intervals."""
    if not intervals:
        return 0.0
    total = 0.0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    total += current_end - current_start
    return total


def _serialize_datetime(value: datetime | None) -> str | None:
    """Normalize timestamps to UTC ISO-8601 for YAML persistence."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()
