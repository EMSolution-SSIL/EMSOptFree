"""
optimization_output_handler.py
The MIT License (MIT)
Copyright © 2025 Science Solutions International Laboratory, Inc.

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

import math
import shutil
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from logging import getLogger
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from emsopt_engine.configs.optimization import OutputConfig
from emsopt_engine.individual import Population
from emsopt_engine.interface.mo_optimizer_base import MOOptimizerBase
from emsopt_engine.interface.optimizer_interface import OptimizerInterface
from emsopt_engine.interface.so_optimizer_base import SOOptimizerBase
from manager.optimization_analyzer import OptimizationAnalyzer
from utils.individuals_io import (
    dump_individuals,
    dump_individuals_csv,
    load_individuals,
    merge_candidate_csv_files,
    normalize_outcome_filepath_for_summary,
    output_outcome,
    rewrite_outcome_filepaths_in_csv,
)

logger = getLogger(__name__)

RUN_RECORDS_DIRNAME = "records"
RUN_RECORDS_FILENAME = "run_records.csv"
RUN_MANIFEST_FILENAME = "run_manifest.yaml"
CROSS_STUDY_DIRNAME = "cross_study"
CROSS_STUDY_TABLE_FILENAME = "cross_study_individuals.csv"
CROSS_STUDY_MANIFEST_FILENAME = "cross_study_manifest.yaml"
SURROGATE_TRAINING_DATA_FILENAME = "surrogate_training_data.csv"
SURROGATE_TRAINING_METADATA_FILENAME = "surrogate_training_data.yaml"

yaml.add_multi_representer(
    np.floating, lambda dumper, value: dumper.represent_float(float(value)), Dumper=yaml.SafeDumper
)
yaml.add_multi_representer(Path, lambda dumper, value: dumper.represent_str(str(value)), Dumper=yaml.SafeDumper)


def _resolve_run_records_csv(summary_dir: Path) -> Path | None:
    """Return the per-run records table exported under a study summary directory."""
    run_records_csv = summary_dir / RUN_RECORDS_DIRNAME / RUN_RECORDS_FILENAME
    if run_records_csv.exists():
        return run_records_csv
    return None


def _rewrite_outcome_filepaths_in_yaml(
    filepath: Path,
    *,
    summary_dir: Path,
    source_root: Path | None,
) -> None:
    if not filepath.exists():
        return
    individuals = load_individuals(str(filepath))
    changed = False
    for individual in individuals:
        normalized = normalize_outcome_filepath_for_summary(
            individual.outcome_filepath,
            summary_dir=summary_dir,
            source_root=source_root,
        )
        if normalized != individual.outcome_filepath:
            individual.outcome_filepath = normalized
            changed = True
    if changed:
        dump_individuals(individuals, str(filepath))


def _is_in_case_dir_artifact(path: Path, *, root: Path) -> bool:
    try:
        relative_parts = path.relative_to(root).parts
    except ValueError:
        return False
    return any(part.endswith("_case_dir") for part in relative_parts[:-1])


def _normalize_persisted_outcome_paths(summary_dir: Path, *, source_root: Path | None) -> None:
    resultant_dir = summary_dir / "resultant"
    if resultant_dir.exists():
        for yaml_path in resultant_dir.rglob("*.yaml"):
            if _is_in_case_dir_artifact(yaml_path, root=resultant_dir):
                continue
            _rewrite_outcome_filepaths_in_yaml(yaml_path, summary_dir=summary_dir, source_root=source_root)
        for csv_path in resultant_dir.rglob("*.csv"):
            if _is_in_case_dir_artifact(csv_path, root=resultant_dir):
                continue
            rewrite_outcome_filepaths_in_csv(csv_path, summary_dir=summary_dir, source_root=source_root)
    run_records_csv = _resolve_run_records_csv(summary_dir)
    if run_records_csv is not None:
        rewrite_outcome_filepaths_in_csv(run_records_csv, summary_dir=summary_dir, source_root=source_root)


def output_case_dir(individual: object, filepath_base: str | Path) -> str | None:
    """Copy an evaluated individual's analysis working directory next to result artifacts."""
    working_dir = getattr(individual, "working_dir", None)
    if not working_dir:
        return None
    source_dir = Path(working_dir)
    if not source_dir.is_dir():
        logger.warning("Case directory does not exist: %s. Skip case save.", source_dir)
        return None

    destination_parent = Path(f"{filepath_base}_case_dir")
    destination_dir = destination_parent / source_dir.name
    source_resolved = source_dir.resolve()
    destination_resolved = destination_dir.resolve(strict=False)
    if source_resolved == destination_resolved or source_resolved in destination_resolved.parents:
        logger.warning("Case directory destination is inside source: %s. Skip case save.", destination_dir)
        return None

    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    destination_parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, destination_dir)
    return str(destination_dir)


class OptimizationOutputHandlerInterface(ABC):
    @abstractmethod
    def output_best_individual(
        self, optimizer: OptimizerInterface, output_dir: Path, i_iter: int, outconf: OutputConfig
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def output_candidate(self, candidates: Population, output_dir: Path, i_iter: int, outconf: OutputConfig) -> None:
        raise NotImplementedError

    @abstractmethod
    def output_candidate_plot(
        self, candidates: Population, output_dir: Path, i_iter: int, outconf: OutputConfig
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_latest_summary_dir(self, summary_dir: str) -> None:
        raise NotImplementedError


class SOHandler(OptimizationOutputHandlerInterface):
    def __init__(
        self, material_color_map: dict[int, str] | None = None, image_export: dict[str, object] | None = None
    ) -> None:
        self.analyzer = OptimizationAnalyzer()
        self.latest_result_dir: str = ""
        self._candidate_csv_files: list[str] = []
        self.material_color_map = dict(material_color_map or {})
        self.image_export = dict(image_export or {})

    def output_best_individual(
        self, optimizer: SOOptimizerBase, output_dir: Path, i_iter: int, outconf: OutputConfig
    ) -> None:
        if optimizer.best_history:
            iter_label = f"iter{i_iter + 1}_"
            out_f = str(output_dir / f"{iter_label}{outconf.filename_base}")
            optimizer.best_history[-1].outcome_filepath = output_outcome(
                optimizer.best_history[-1],
                out_f,
                material_color_map=self.material_color_map,
                image_export=self.image_export,
            )
            if getattr(outconf, "save_case_dir", False) is True:
                output_case_dir(optimizer.best_history[-1], out_f)
            yaml_f = output_dir / outconf.filename_base
            dump_individuals(optimizer.best_history, str(yaml_f.with_suffix(".yaml")))
            dump_individuals_csv(optimizer.best_history, str(yaml_f.with_suffix(".csv")))
        self.latest_result_dir = str(output_dir)

    def output_candidate(self, candidates: Population, output_dir: Path, i_iter: int, outconf: OutputConfig) -> None:
        if len(candidates) <= 0:
            return
        iter_label = f"iter{i_iter + 1}_"
        for i in range(len(candidates)):
            out_f = str(output_dir / f"{iter_label}{i + 1}_{outconf.filename_base}")
            candidates[i].outcome_filepath = output_outcome(
                candidates[i], out_f, material_color_map=self.material_color_map, image_export=self.image_export
            )
            if getattr(outconf, "save_case_dir", False) is True:
                output_case_dir(candidates[i], out_f)
        yaml_f = output_dir / f"{iter_label}{outconf.filename_base}"
        dump_individuals(list(candidates.values()), str(yaml_f.with_suffix(".yaml")))
        csv_path = str(yaml_f.with_suffix(".csv"))
        dump_individuals_csv(list(candidates.values()), csv_path)
        self._candidate_csv_files.append(csv_path)
        if getattr(outconf, "save_case_dir", False) is True:
            self.latest_result_dir = str(output_dir)

    def output_candidate_plot(
        self, candidates: Population, output_dir: Path, i_iter: int, outconf: OutputConfig
    ) -> None:
        if len(candidates) <= 0:
            return
        if self.analyzer.pca is None:
            self.analyzer.fit_pca(candidates)
        plot_f = output_dir / f"iter{i_iter + 1}_{outconf.filename_base}"
        self.analyzer.plot_population(candidates, str(plot_f.with_suffix(".png")))

    def save_latest_summary_dir(self, summary_dir: str) -> None:
        summary_path = Path(summary_dir)
        resultant_dir = summary_path / "resultant"
        records_dir = summary_path / RUN_RECORDS_DIRNAME
        if self.latest_result_dir != "":
            shutil.copytree(self.latest_result_dir, resultant_dir)
        merge_candidate_csv_files(self._candidate_csv_files, str(records_dir), RUN_RECORDS_FILENAME)
        _normalize_persisted_outcome_paths(
            summary_path, source_root=Path(self.latest_result_dir) if self.latest_result_dir else None
        )
        _write_run_manifest(records_dir)
        output_surrogate_result_artifacts(str(resultant_dir), source_csv=_resolve_run_records_csv(summary_path))


class MOHandler(OptimizationOutputHandlerInterface):
    def __init__(
        self, material_color_map: dict[int, str] | None = None, image_export: dict[str, object] | None = None
    ) -> None:
        self.analyzer = OptimizationAnalyzer()
        self.latest_result_dir: str = ""
        self._candidate_csv_files: list[str] = []
        self.material_color_map = dict(material_color_map or {})
        self.image_export = dict(image_export or {})

    def output_best_individual(
        self, optimizer: MOOptimizerBase, output_dir: Path, i_iter: int, outconf: OutputConfig
    ) -> None:
        if optimizer.pareto:
            output_subdir = output_dir / f"iter{i_iter + 1}"
            output_subdir.mkdir(exist_ok=True)
            for i, ind in enumerate(optimizer.pareto.values()):
                out_f = str(output_subdir / f"pareto{i + 1}_{outconf.filename_base}")
                ind.outcome_filepath = output_outcome(
                    ind, out_f, material_color_map=self.material_color_map, image_export=self.image_export
                )
                if getattr(outconf, "save_case_dir", False) is True:
                    output_case_dir(ind, out_f)
            yaml_f = output_subdir / outconf.filename_base
            dump_individuals(list(optimizer.pareto.values()), str(yaml_f.with_suffix(".yaml")))
            dump_individuals_csv(list(optimizer.pareto.values()), str(yaml_f.with_suffix(".csv")))
            self.latest_result_dir = str(output_subdir)

    def output_candidate(self, candidates: Population, output_dir: Path, i_iter: int, outconf: OutputConfig) -> None:
        if len(candidates) <= 0:
            return
        iter_label = f"iter{i_iter + 1}_"
        for i in range(len(candidates)):
            out_f = str(output_dir / f"{iter_label}{i + 1}_{outconf.filename_base}")
            candidates[i].outcome_filepath = output_outcome(
                candidates[i], out_f, material_color_map=self.material_color_map, image_export=self.image_export
            )
            if getattr(outconf, "save_case_dir", False) is True:
                output_case_dir(candidates[i], out_f)
        yaml_f = output_dir / f"{iter_label}{outconf.filename_base}"
        dump_individuals(list(candidates.values()), str(yaml_f.with_suffix(".yaml")))
        csv_path = str(yaml_f.with_suffix(".csv"))
        dump_individuals_csv(list(candidates.values()), csv_path)
        self._candidate_csv_files.append(csv_path)
        if getattr(outconf, "save_case_dir", False) is True:
            self.latest_result_dir = str(output_dir)

    def output_candidate_plot(
        self, candidates: Population, output_dir: Path, i_iter: int, outconf: OutputConfig
    ) -> None:
        if len(candidates) <= 0:
            return
        if self.analyzer.pca is None:
            self.analyzer.fit_pca(candidates)
        plot_f = output_dir / f"iter{i_iter + 1}_{outconf.filename_base}"
        self.analyzer.plot_population(candidates, str(plot_f.with_suffix(".png")))

    def save_latest_summary_dir(self, summary_dir: str) -> None:
        summary_path = Path(summary_dir)
        resultant_dir = summary_path / "resultant"
        records_dir = summary_path / RUN_RECORDS_DIRNAME
        if self.latest_result_dir != "":
            shutil.copytree(self.latest_result_dir, resultant_dir)
        merge_candidate_csv_files(self._candidate_csv_files, str(records_dir), RUN_RECORDS_FILENAME)
        _normalize_persisted_outcome_paths(
            summary_path, source_root=Path(self.latest_result_dir) if self.latest_result_dir else None
        )
        _write_run_manifest(records_dir)
        output_surrogate_result_artifacts(str(resultant_dir), source_csv=_resolve_run_records_csv(summary_path))


def _collect_prediction_pairs(row: pd.Series) -> list[tuple[float, float]]:
    predicted_cols = sorted([c for c in row.index if c.startswith("surrogate_predicted_value_")])
    true_cols = sorted([c for c in row.index if c.startswith("surrogate_true_value_")])
    if not predicted_cols and "surrogate_predicted_value" in row.index and "surrogate_true_value" in row.index:
        predicted_cols = ["surrogate_predicted_value"]
        true_cols = ["surrogate_true_value"]
    pairs = []
    for pred_col, true_col in zip(predicted_cols, true_cols, strict=False):
        pred_value = row.get(pred_col)
        true_value = row.get(true_col)
        if pd.isna(pred_value) or pd.isna(true_value):
            continue
        pairs.append((float(pred_value), float(true_value)))
    return pairs


def build_surrogate_summary(df: pd.DataFrame) -> dict[str, object] | None:
    surrogate_cols = [col for col in df.columns if col.startswith("surrogate_")]
    if not surrogate_cols:
        return None
    summary: dict[str, object] = {"num_individuals": len(df)}
    if "surrogate_evaluation_method" in df.columns:
        counts = df["surrogate_evaluation_method"].dropna().astype(str).value_counts().to_dict()
        summary["evaluation_method_counts"] = {str(key): int(value) for key, value in counts.items()}
    for col, prefix in [
        ("surrogate_surrogate_used", "surrogate_usage"),
        ("surrogate_true_evaluation_used", "true_evaluation"),
        ("surrogate_inference_failed", "inference_failure"),
    ]:
        if col in df.columns:
            values = df[col].fillna(value=False).astype(bool)
            count = int(values.sum())
            summary[f"{prefix}_count"] = count
            summary[f"{prefix}_rate"] = 0.0 if len(values) == 0 else count / len(values)
        if "surrogate_fallback_reason" in df.columns:
            reasons = df["surrogate_fallback_reason"].dropna().astype(str)
            if not reasons.empty:
                summary["fallback_reason_counts"] = {
                    str(key): int(value) for key, value in reasons.value_counts().to_dict().items()
                }
    abs_errors = []
    signed_errors = []
    for _, row in df.iterrows():
        for predicted, true_value in _collect_prediction_pairs(row):
            signed_error = predicted - true_value
            signed_errors.append(signed_error)
            abs_errors.append(abs(signed_error))
    if abs_errors:
        squared = [err**2 for err in signed_errors]
        summary["prediction_error_sample_count"] = len(abs_errors)
        summary["prediction_error_mae"] = float(sum(abs_errors) / len(abs_errors))
        summary["prediction_error_rmse"] = float(math.sqrt(sum(squared) / len(squared)))
        summary["prediction_error_max_abs"] = float(max(abs_errors))
        summary["prediction_error_mean_signed"] = float(sum(signed_errors) / len(signed_errors))
    return summary


def output_surrogate_result_artifacts(
    dst_dir: str, source_csv: str | Path | None = None, metadata: dict[str, object] | None = None
) -> None:
    """Export surrogate-related CSV/YAML artifacts from a merged candidate table."""
    base_dir = Path(dst_dir)
    resolved_source_csv = Path(source_csv) if source_csv is not None else None
    if resolved_source_csv is None or not resolved_source_csv.exists():
        return
    df = pd.read_csv(resolved_source_csv)
    surrogate_cols = [col for col in df.columns if col.startswith("surrogate_")]
    if not surrogate_cols:
        return
    context_cols = [
        col for col in df.columns if col in {"fitness", "constraint_violation"} or col.startswith("objective_")
    ]
    usage_df = df[[*context_cols, *surrogate_cols]]
    usage_df.to_csv(base_dir / "surrogate_individuals.csv", index=False)
    summary = build_surrogate_summary(df)
    if summary is not None:
        if metadata:
            summary.update(metadata)
        with (base_dir / "surrogate_summary.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(summary, f, sort_keys=False)


def _get_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name that exists in the DataFrame."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def build_cross_study_records(
    df: pd.DataFrame, project_name: str, study_name: str, run_id: str | None = None
) -> pd.DataFrame:
    """Attach project/study/run identifiers to per-run candidate records."""
    enriched_df = df.copy()
    resolved_run_id = run_id or f"{study_name}_run"
    status_col = _get_first_existing_column(
        enriched_df, ["record_status", "status", "evaluation_status", "analysis_status"]
    )
    failure_col = _get_first_existing_column(
        enriched_df, ["record_failure_reason", "failure_reason", "analysis_failure_reason"]
    )
    if status_col is None:
        enriched_df["record_status"] = "success"
    elif status_col != "record_status":
        enriched_df["record_status"] = enriched_df[status_col]
    if failure_col is None:
        enriched_df["record_failure_reason"] = pd.Series([None] * len(enriched_df), dtype="object")
    elif failure_col != "record_failure_reason":
        enriched_df["record_failure_reason"] = enriched_df[failure_col]
    enriched_df.insert(
        0,
        "record_id",
        [f"{project_name}:{study_name}:{resolved_run_id}:{idx}" for idx in range(len(enriched_df))],
    )
    enriched_df.insert(0, "run_id", resolved_run_id)
    enriched_df.insert(0, "study_id", study_name)
    enriched_df.insert(0, "project_id", project_name)
    return enriched_df


def append_cross_study_table(
    project_summary_dir: str, study_summary_dir: str, project_name: str, study_name: str, run_id: str | None = None
) -> Path | None:
    """Append one study result into the project-level cross-study table."""
    study_summary_path = Path(study_summary_dir)
    source_csv = _resolve_run_records_csv(study_summary_path)
    if source_csv is None:
        return None

    cross_study_dir = Path(project_summary_dir) / CROSS_STUDY_DIRNAME
    cross_study_dir.mkdir(parents=True, exist_ok=True)
    cross_study_csv = cross_study_dir / CROSS_STUDY_TABLE_FILENAME

    current_df = pd.read_csv(source_csv)
    enriched_df = build_cross_study_records(current_df, project_name=project_name, study_name=study_name, run_id=run_id)
    if cross_study_csv.exists():
        existing_df = pd.read_csv(cross_study_csv)
        enriched_df = pd.concat([existing_df, enriched_df], ignore_index=True)
        enriched_df = enriched_df.drop_duplicates(subset=["record_id"], keep="last")
    enriched_df.to_csv(cross_study_csv, index=False)
    _write_cross_study_manifest(cross_study_dir, enriched_df)
    return cross_study_csv


def _write_run_manifest(records_dir: Path) -> Path | None:
    """Write metadata for the run-level records table."""
    run_records_csv = records_dir / RUN_RECORDS_FILENAME
    if not run_records_csv.exists():
        return None

    df = pd.read_csv(run_records_csv)
    manifest = {
        "schema_version": 1,
        "table_filename": RUN_RECORDS_FILENAME,
        "num_rows": len(df),
        "status_counts": {
            str(key): int(value)
            for key, value in df["record_status"].fillna("unknown").value_counts().to_dict().items()
        }
        if "record_status" in df.columns
        else {},
        "updated_at": datetime.now(UTC).isoformat(),
    }
    manifest_path = records_dir / RUN_MANIFEST_FILENAME
    with manifest_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    return manifest_path


def _write_cross_study_manifest(cross_study_dir: Path, df: pd.DataFrame) -> Path:
    """Write metadata for the project-level cross-study table."""
    manifest = {
        "schema_version": 1,
        "table_filename": CROSS_STUDY_TABLE_FILENAME,
        "num_rows": len(df),
        "study_ids": sorted(df["study_id"].dropna().astype(str).unique().tolist()) if "study_id" in df.columns else [],
        "status_counts": {
            str(key): int(value)
            for key, value in df["record_status"].fillna("unknown").value_counts().to_dict().items()
        }
        if "record_status" in df.columns
        else {},
        "updated_at": datetime.now(UTC).isoformat(),
    }
    manifest_path = cross_study_dir / CROSS_STUDY_MANIFEST_FILENAME
    with manifest_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    return manifest_path


def export_surrogate_training_data(
    project_summary_dir: str,
    *,
    study_names: list[str] | None = None,
    statuses: list[str] | None = None,
) -> Path | None:
    """Export a filtered training dataset from the project-level cross-study table."""
    cross_study_dir = Path(project_summary_dir) / CROSS_STUDY_DIRNAME
    cross_study_csv = cross_study_dir / CROSS_STUDY_TABLE_FILENAME
    if not cross_study_csv.exists():
        return None

    df = pd.read_csv(cross_study_csv)
    filtered_df = df.copy()
    if study_names is not None:
        filtered_df = filtered_df[filtered_df["study_id"].isin(study_names)]
    if statuses is not None and "record_status" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["record_status"].isin(statuses)]

    output_csv = cross_study_dir / SURROGATE_TRAINING_DATA_FILENAME
    filtered_df.to_csv(output_csv, index=False)
    metadata = {
        "source_study_ids": sorted(filtered_df["study_id"].dropna().astype(str).unique().tolist())
        if "study_id" in filtered_df.columns
        else [],
        "filter_conditions": {"study_ids": study_names, "statuses": statuses},
        "num_rows": len(filtered_df),
    }
    with (cross_study_dir / SURROGATE_TRAINING_METADATA_FILENAME).open("w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)
    return output_csv
