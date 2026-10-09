"""
individuals_io.py
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

import contextlib
import shutil
from dataclasses import asdict
from logging import getLogger
from pathlib import Path

import meshio
import numpy as np
import pandas as pd
import pyvista as pv
import yaml

from emsopt_engine.configs.base import BaseConfig
from emsopt_engine.individual import EvaluationRecord, Individual, OptimizationProblemMetrics

logger = getLogger(__name__)

SURROGATE_INFO_ATTRIBUTE = "surrogate_info"
RECORD_INFO_ATTRIBUTE = "record_info"
LABEL_VALUE_PREFIX = "label__"
OUTCOME_FILEPATH_COLUMN = "outcome_filepath"


# add yaml representer
yaml.add_multi_representer(
    np.floating, lambda dumper, value: dumper.represent_float(float(value)), Dumper=yaml.SafeDumper
)
yaml.add_representer(np.ndarray, lambda dumper, value: dumper.represent_list(value.tolist()))
yaml.add_multi_representer(Path, lambda dumper, value: dumper.represent_str(str(value)), Dumper=yaml.SafeDumper)


def merge_candidate_csv_files(
    csv_files: list[str], dst_dir: str, output_filename: str = "merged_candidates.csv"
) -> None:
    """
    merge files in csv_files and save them to dst_dir/output_filename
    Args:
        csv_files (list[str]): file pathes
        dst_dir (str): destination directory
        output_filename (str): destination csv filename
    """
    import pandas as pd

    dfs = []
    for csv_path in csv_files:
        try:
            dfs.append(pd.read_csv(csv_path))
        except (FileNotFoundError, pd.errors.ParserError, ValueError) as e:
            logger.warning("Failed to read %s: %s", csv_path, e)
    if dfs:
        Path(dst_dir).mkdir(exist_ok=True)
        dst_csv = Path(dst_dir) / output_filename
        # if dst_csv already exists, merge them
        if dst_csv.exists():
            dfs.append(pd.read_csv(dst_csv))
        merged_df = pd.concat(dfs, ignore_index=True)
        merged_df.to_csv(dst_csv, index=False)


def _normalize_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _normalize_sequence(value: object) -> list[object]:
    if isinstance(value, np.ndarray):
        return [_normalize_scalar(v) for v in value.flatten().tolist()]
    if isinstance(value, (list, tuple)):
        return [_normalize_scalar(v) for v in value]
    return [_normalize_scalar(value)]


def _get_metadata_raw_info(individual: Individual, attribute_name: str) -> dict[str, object]:
    """Fetch optional dict-like metadata attached to an individual."""
    raw = getattr(individual, attribute_name, None)
    if isinstance(raw, dict):
        return dict(raw)
    return {}


def _get_surrogate_raw_info(individual: Individual) -> dict[str, object]:
    return _get_metadata_raw_info(individual, SURROGATE_INFO_ATTRIBUTE)


def _load_record_info(raw: dict[str, object] | None) -> EvaluationRecord:
    if raw is None:
        return EvaluationRecord()
    status = str(raw.get("status", "success"))
    failure_reason = raw.get("failure_reason")
    if failure_reason is not None:
        failure_reason = str(failure_reason)
    return EvaluationRecord(status=status, failure_reason=failure_reason)


def extract_record_info(individual: Individual) -> dict[str, object]:
    """Normalize per-individual execution metadata for serialization and CSV output."""
    return {key: _normalize_scalar(value) for key, value in vars(individual.record_info).items() if value is not None}


def _normalize_surrogate_fields(info: dict[str, object]) -> None:
    for field in ("surrogate_used", "true_evaluation_used", "inference_failed"):
        if field in info:
            info[field] = bool(info[field])
    for field in ("predicted_value", "true_value"):
        if field in info:
            info[field] = _normalize_sequence(info[field])


def _infer_evaluation_method(info: dict[str, object]) -> None:
    if "evaluation_method" in info:
        return
    surrogate_used = bool(info.get("surrogate_used", False))
    true_evaluation_used = bool(info.get("true_evaluation_used", False))
    if surrogate_used and true_evaluation_used:
        info["evaluation_method"] = "surrogate_then_true"
    elif surrogate_used:
        info["evaluation_method"] = "surrogate"
    elif true_evaluation_used:
        info["evaluation_method"] = "true"


def extract_surrogate_info(individual: Individual) -> dict[str, object]:
    info = _get_surrogate_raw_info(individual)
    _normalize_surrogate_fields(info)
    _infer_evaluation_method(info)
    return {key: _normalize_scalar(value) for key, value in info.items() if value is not None}


def _serialize_individual(individual: Individual) -> dict[str, object]:
    payload = asdict(individual)
    surrogate_info = extract_surrogate_info(individual)
    if surrogate_info:
        payload[SURROGATE_INFO_ATTRIBUTE] = surrogate_info
    record_info = extract_record_info(individual)
    if record_info:
        payload[RECORD_INFO_ATTRIBUTE] = record_info
    return payload


def _expand_surrogate_info(info: dict[str, object]) -> dict[str, object]:
    expanded: dict[str, object] = {}
    for key, value in info.items():
        column_base = f"surrogate_{key}"
        if isinstance(value, list):
            for idx, elem in enumerate(value, start=1):
                expanded[f"{column_base}_{idx}"] = elem
        else:
            expanded[column_base] = value
    return expanded


def _expand_label_values(individual: Individual) -> dict[str, object]:
    label_values = getattr(individual, "label_values", {})
    return {f"{LABEL_VALUE_PREFIX}{label}": value for label, value in label_values.items()}


def _to_posix_relative_path(path: Path) -> str:
    return path.as_posix()


def _normalize_relative_outcome_filepath(path: Path, *, summary_root: Path) -> str:
    normalized = Path(path.as_posix())
    resultant_dirname = BaseConfig.RESULTANT_DIRNAME.value
    if normalized.parts and normalized.parts[0] == resultant_dirname:
        return _to_posix_relative_path(normalized)
    if (summary_root / normalized).exists():
        return _to_posix_relative_path(normalized)
    legacy_candidate = summary_root / resultant_dirname / normalized.name
    if legacy_candidate.exists():
        return _to_posix_relative_path(Path(resultant_dirname) / normalized.name)
    return _to_posix_relative_path(normalized)


def _normalize_absolute_outcome_filepath(
    path: Path,
    *,
    summary_root: Path,
    source_root: Path | None,
) -> str | None:
    resultant_prefix = Path(BaseConfig.RESULTANT_DIRNAME.value)
    if source_root is not None:
        try:
            relative_to_source = path.relative_to(source_root)
        except ValueError:
            pass
        else:
            return _to_posix_relative_path(resultant_prefix / relative_to_source)

    try:
        relative_to_summary = path.relative_to(summary_root)
    except ValueError:
        pass
    else:
        return _to_posix_relative_path(relative_to_summary)

    return _find_summary_relative_outcome_match(path, summary_root=summary_root)


def _find_summary_relative_outcome_match(path: Path, *, summary_root: Path) -> str | None:
    resultant_dir = summary_root / BaseConfig.RESULTANT_DIRNAME.value
    matches = [candidate for candidate in resultant_dir.rglob(path.name) if candidate.is_file()]
    if len(matches) == 1:
        return _to_posix_relative_path(matches[0].relative_to(summary_root))
    return None


def normalize_outcome_filepath_for_summary(
    outcome_filepath: str | None,
    *,
    summary_dir: str | Path,
    source_root: str | Path | None = None,
) -> str | None:
    """Normalize an outcome path so it can be persisted relative to the run summary directory."""
    if not outcome_filepath:
        return None

    summary_root = Path(summary_dir)
    path = Path(outcome_filepath)
    if not path.is_absolute():
        return _normalize_relative_outcome_filepath(path, summary_root=summary_root)
    return _normalize_absolute_outcome_filepath(
        path,
        summary_root=summary_root,
        source_root=Path(source_root) if source_root is not None else None,
    )


def resolve_outcome_filepath(outcome_filepath: str | None, *, summary_dir: str | Path) -> Path | None:
    """Resolve a persisted outcome path against a run summary directory."""
    if not outcome_filepath:
        return None

    summary_root = Path(summary_dir)
    path = Path(outcome_filepath)
    if path.is_absolute():
        return path

    normalized = Path(path.as_posix())
    direct_candidate = summary_root / normalized
    if direct_candidate.exists():
        return direct_candidate

    legacy_candidate = summary_root / BaseConfig.RESULTANT_DIRNAME.value / normalized.name
    if legacy_candidate.exists():
        return legacy_candidate
    return direct_candidate


def rewrite_outcome_filepaths_in_csv(
    filepath: str | Path,
    *,
    summary_dir: str | Path,
    source_root: str | Path | None = None,
) -> None:
    """Rewrite the outcome filepath column in an existing CSV without touching other columns."""
    csv_path = Path(filepath)
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if OUTCOME_FILEPATH_COLUMN not in df.columns:
        return
    df[OUTCOME_FILEPATH_COLUMN] = [
        normalize_outcome_filepath_for_summary(
            None if pd.isna(value) else str(value),
            summary_dir=summary_dir,
            source_root=source_root,
        )
        for value in df[OUTCOME_FILEPATH_COLUMN]
    ]
    df.to_csv(csv_path, index=False)


def dump_individuals(individuals: list[Individual], filepath: str) -> None:
    """Dump individuals to yaml file using yaml.safe_dump

    Args:
        individuals (list[Individual]): individuals to dump
        filepath (str): dst filepath
    """
    dicts = [_serialize_individual(ind) for ind in individuals]
    with Path(filepath).open(mode="w", encoding="utf-8") as f:
        yaml.safe_dump(dicts, f)


def dump_individuals_csv(individuals: list[Individual], filepath: str) -> None:
    """Dump individuals to csv file using yaml.safe_dump

    Args:
        individuals (list[Individual]): individuals to dump
        filepath (str): dst filepath
    """
    # convert to dataframe-like list
    records = []
    for ind in individuals:
        d_expanded = {
            "fitness": ind.metrics.fitness,
            "constraint_violation": ind.metrics.constraint_violation,
            OUTCOME_FILEPATH_COLUMN: ind.outcome_filepath,
            **{f"objective_{i + 1}": v for i, v in enumerate(ind.metrics.objectives)},
            **{f"ineq_constraint_{i + 1}": v for i, v in enumerate(ind.metrics.ineq_constraints)},
            **{f"eq_contraint_{i + 1}": v for i, v in enumerate(ind.metrics.eq_constraints)},
            **{f"other_metrics_{i + 1}": v for i, v in enumerate(ind.metrics.other_metrics)},
            **{f"solution_{i + 1}": v for i, v in enumerate(ind.solution)},
        }
        d_expanded.update(_expand_label_values(ind))
        d_expanded.update(_expand_surrogate_info(extract_surrogate_info(ind)))
        d_expanded.update({f"record_{key}": value for key, value in extract_record_info(ind).items()})
        records.append(d_expanded)
    # create pandas DataFrame and convert it to csv
    df = pd.DataFrame(records)
    df.to_csv(filepath, index=False)


def load_individuals(filepath: str) -> list[Individual]:
    """Load individuals from yaml file using yaml.safe_load

    Args:
        filepath (str): src filepath

    Returns:
        list[Individual]: individuals loaded
    """
    with Path(filepath).open(mode="r", encoding="utf-8") as f:
        dicts = yaml.safe_load(f)
    individuals = []
    for d in dicts:
        metrics = OptimizationProblemMetrics(**d["metrics"])
        individual = Individual(
            solution=d["solution"],
            outcome_filepath=d["outcome_filepath"],
            metrics=metrics,
            label_values=d.get("label_values", {}),
            record_info=_load_record_info(d.get(RECORD_INFO_ATTRIBUTE)),
        )
        surrogate_info = d.get(SURROGATE_INFO_ATTRIBUTE)
        if isinstance(surrogate_info, dict):
            setattr(individual, SURROGATE_INFO_ATTRIBUTE, surrogate_info)
        individuals.append(individual)
    return individuals


def load_individuals_csv(filepath: str) -> list[Individual]:
    """Load individuals from csv file

    Args:
        filepath (str): src filepath

    Returns:
        list[Individual]: individuals loaded
    """
    df = pd.read_csv(filepath)
    objectives = df[[col for col in df.columns if col.startswith("objective_")]].to_numpy()
    ineq_cons = df[[col for col in df.columns if col.startswith("ineq_constraint_")]].to_numpy()
    eq_cons = df[[col for col in df.columns if col.startswith("eq_constraint_")]].to_numpy()
    others = df[[col for col in df.columns if col.startswith("other_metrics_")]].to_numpy()
    solutions = df[[col for col in df.columns if col.startswith("solution_")]].to_numpy()
    label_columns = [col for col in df.columns if col.startswith(LABEL_VALUE_PREFIX)]
    record_status_col = "record_status" if "record_status" in df.columns else None
    record_failure_reason_col = "record_failure_reason" if "record_failure_reason" in df.columns else None
    outcome_col = OUTCOME_FILEPATH_COLUMN if OUTCOME_FILEPATH_COLUMN in df.columns else None
    individuals = []
    for i in range(solutions.shape[0]):
        metrics = OptimizationProblemMetrics(
            objectives=objectives[i].tolist(),
            ineq_constraints=ineq_cons[i].tolist(),
            eq_constraints=eq_cons[i].tolist(),
            other_metrics=others[i].tolist(),
        )
        if "fitness" in df.columns and not pd.isna(df.iloc[i]["fitness"]):
            metrics.fitness = float(df.iloc[i]["fitness"])
        individuals.append(
            Individual(
                solution=solutions[i].tolist(),
                outcome_filepath=(
                    str(df.iloc[i][outcome_col]) if outcome_col and not pd.isna(df.iloc[i][outcome_col]) else None
                ),
                metrics=metrics,
                label_values={
                    col.removeprefix(LABEL_VALUE_PREFIX): float(df.iloc[i][col])
                    for col in label_columns
                    if not pd.isna(df.iloc[i][col])
                },
                record_info=EvaluationRecord(
                    status=str(df.iloc[i][record_status_col])
                    if record_status_col and not pd.isna(df.iloc[i][record_status_col])
                    else "success",
                    failure_reason=(
                        str(df.iloc[i][record_failure_reason_col])
                        if record_failure_reason_col and not pd.isna(df.iloc[i][record_failure_reason_col])
                        else None
                    ),
                ),
            )
        )
    return individuals


# Material ID thresholds for color mapping
MATERIAL_ID_THRESHOLD_AIR = 600000
MATERIAL_ID_THRESHOLD_MAGNET = 100000
MATERIAL_ID_THRESHOLD_COIL = 50000
MATERIAL_ID_THRESHOLD_CORE = 20
MATERIAL_ID_THRESHOLD_LOW = 10


def _convert_msh_to_vtk_celldata(
    mesh: meshio.Mesh, cell_masks: list[np.ndarray | None] | None = None
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Convert meshio Mesh object to VTK cell data arrays for pyvista UnstructuredGrid."""
    all_cells = []
    all_celltypes = []
    cell_data_blocks: dict[str, list[np.ndarray]] = {}
    vtk_type_map = {
        "triangle": 5,  # VTK_TRIANGLE
        "quad": 9,  # VTK_QUAD
    }
    for i, cell_block in enumerate(mesh.cells):
        ctype = cell_block.type
        if ctype not in ["triangle", "quad"]:
            continue
        c_data = cell_block.data
        if cell_masks is not None and cell_masks[i] is not None:
            c_data = c_data[cell_masks[i]]
        if c_data.size == 0:
            continue
        n = c_data.shape[1]
        cell_with_size = np.hstack([np.full((c_data.shape[0], 1), n), c_data])
        all_cells.append(cell_with_size.flatten())
        all_celltypes.append(np.full(c_data.shape[0], vtk_type_map[ctype]))
        for data_name, data_blocks in mesh.cell_data.items():
            data_block = np.asarray(data_blocks[i])
            if cell_masks is not None and cell_masks[i] is not None:
                data_block = data_block[cell_masks[i]]
            cell_data_blocks.setdefault(data_name, []).append(data_block)
    if not all_cells:
        msg = "No triangle or quad cells to visualize"
        raise ValueError(msg)
    cells_vtk = np.concatenate(all_cells).astype(np.int64)
    celltypes_vtk = np.concatenate(all_celltypes)
    cell_data_vtk = {data_name: np.concatenate(data_blocks) for data_name, data_blocks in cell_data_blocks.items()}
    return cells_vtk, celltypes_vtk, cell_data_vtk


def _cell_centroid_angles_deg(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    centroids = np.mean(points[cells, :2], axis=1)
    return (np.degrees(np.arctan2(centroids[:, 1], centroids[:, 0])) + 360.0) % 360.0


def _normalize_image_target_region(
    image_target_region: tuple[tuple[float, float], tuple[float, float]] | list[list[float]] | None,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if image_target_region is None:
        return None
    if len(image_target_region) != 2:
        msg = f"image_export.image_target_region must have 2 axes, but got {len(image_target_region)}"
        raise ValueError(msg)
    axes = []
    for axis_index, axis in enumerate(image_target_region):
        if len(axis) != 2:
            msg = f"image_export.image_target_region axis {axis_index} must have 2 values"
            raise ValueError(msg)
        lower = float(axis[0])
        upper = float(axis[1])
        if np.isclose(lower, upper):
            msg = f"image_export.image_target_region axis {axis_index} must have different lower and upper values"
            raise ValueError(msg)
        axes.append((min(lower, upper), max(lower, upper)))
    return (axes[0], axes[1])


def _cell_centroids_xy(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    return np.mean(points[cells, :2], axis=1)


def _cell_centroids_polar(points: np.ndarray, cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroids = _cell_centroids_xy(points, cells)
    radius = np.linalg.norm(centroids, axis=1)
    theta = (np.degrees(np.arctan2(centroids[:, 1], centroids[:, 0])) + 360.0) % 360.0
    return radius, theta


def _build_image_target_region_cell_masks(
    mesh: meshio.Mesh, *, view_mode: str, image_target_region: tuple[tuple[float, float], tuple[float, float]]
) -> list[np.ndarray | None]:
    masks: list[np.ndarray | None] = []
    points = np.asarray(mesh.points, dtype=float)
    for cell_block in mesh.cells:
        if cell_block.type not in {"triangle", "quad"}:
            masks.append(None)
            continue
        if view_mode in {"cartesian_full", "cartesian_aspect_fit"}:
            x_bounds, y_bounds = image_target_region
            centroids = _cell_centroids_xy(points, cell_block.data)
            mask = (
                (centroids[:, 0] >= x_bounds[0])
                & (centroids[:, 0] <= x_bounds[1])
                & (centroids[:, 1] >= y_bounds[0])
                & (centroids[:, 1] <= y_bounds[1])
            )
        else:
            r_bounds, theta_bounds = image_target_region
            radius, theta = _cell_centroids_polar(points, cell_block.data)
            mask = (
                (radius >= r_bounds[0])
                & (radius <= r_bounds[1])
                & (theta >= theta_bounds[0])
                & (theta <= theta_bounds[1])
            )
        masks.append(mask)
    return masks


def _build_view_cell_masks(
    mesh: meshio.Mesh,
    *,
    view_mode: str,
    theta_max: float,
    image_target_region: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> list[np.ndarray | None] | None:
    if image_target_region is not None:
        return _build_image_target_region_cell_masks(mesh, view_mode=view_mode, image_target_region=image_target_region)
    if view_mode in {"cartesian_full", "cartesian_aspect_fit"}:
        return None
    masks: list[np.ndarray | None] = []
    points = np.asarray(mesh.points, dtype=float)
    for cell_block in mesh.cells:
        if cell_block.type not in {"triangle", "quad"}:
            masks.append(None)
            continue
        angles = _cell_centroid_angles_deg(points, cell_block.data)
        masks.append((angles >= 0.0) & (angles <= theta_max))
    return masks


def _referenced_point_indices(mesh: meshio.Mesh, cell_masks: list[np.ndarray | None] | None = None) -> np.ndarray:
    indices = []
    for i, cell_block in enumerate(mesh.cells):
        if cell_block.type not in {"triangle", "quad"}:
            continue
        cells = cell_block.data
        if cell_masks is not None and cell_masks[i] is not None:
            cells = cells[cell_masks[i]]
        if cells.size:
            indices.append(cells.reshape(-1))
    if not indices:
        msg = "No referenced triangle or quad points to visualize"
        raise ValueError(msg)
    return np.unique(np.concatenate(indices))


def _scale_values_to_unit_interval(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    if np.isclose(upper, lower):
        return np.full_like(values, 0.5, dtype=float)
    return (values - lower) / (upper - lower)


def _scale_cartesian_with_aspect_fit(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    referenced: np.ndarray,
    *,
    x_bounds: tuple[float, float] | None = None,
    y_bounds: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x_min, x_max = (
        x_bounds if x_bounds is not None else (float(np.min(x_raw[referenced])), float(np.max(x_raw[referenced])))
    )
    y_min, y_max = (
        y_bounds if y_bounds is not None else (float(np.min(y_raw[referenced])), float(np.max(y_raw[referenced])))
    )
    x_span = x_max - x_min
    y_span = y_max - y_min
    scale = max(x_span, y_span)
    if np.isclose(scale, 0.0):
        return np.full_like(x_raw, 0.5, dtype=float), np.full_like(y_raw, 0.5, dtype=float)

    x_margin = (1.0 - x_span / scale) * 0.5
    y_margin = (1.0 - y_span / scale) * 0.5
    x_scaled = (x_raw - x_min) / scale + x_margin
    y_scaled = (y_raw - y_min) / scale + y_margin
    return x_scaled, y_scaled


def _image_theta_max(sym_deg: float, num_rotate: int) -> float:
    return float(2.0 * sym_deg * (num_rotate + 1))


def _prepare_image_points_and_masks(
    mesh: meshio.Mesh,
    *,
    view_mode: str,
    sym_deg: float,
    num_rotate: int,
    image_target_region: tuple[tuple[float, float], tuple[float, float]] | list[list[float]] | None = None,
) -> tuple[np.ndarray, list[np.ndarray | None] | None]:
    if view_mode not in {"cartesian_full", "cartesian_aspect_fit", "polar_full", "polar_symmetry"}:
        msg = f"Unsupported image_export view_mode: {view_mode}"
        raise ValueError(msg)

    normalized_region = _normalize_image_target_region(image_target_region)
    points = np.asarray(mesh.points, dtype=float).copy()
    theta_max = _image_theta_max(sym_deg, num_rotate) if view_mode == "polar_full" else float(sym_deg)
    cell_masks = _build_view_cell_masks(
        mesh, view_mode=view_mode, theta_max=theta_max, image_target_region=normalized_region
    )
    referenced = _referenced_point_indices(mesh, cell_masks)

    if view_mode == "cartesian_full":
        x_raw = points[:, 0]
        y_raw = points[:, 1]
        if normalized_region is None:
            x_bounds = (float(np.min(x_raw[referenced])), float(np.max(x_raw[referenced])))
            y_bounds = (float(np.min(y_raw[referenced])), float(np.max(y_raw[referenced])))
        else:
            x_bounds, y_bounds = normalized_region
        points[:, 0] = _scale_values_to_unit_interval(x_raw, *x_bounds)
        points[:, 1] = _scale_values_to_unit_interval(y_raw, *y_bounds)
    elif view_mode == "cartesian_aspect_fit":
        x_raw = points[:, 0]
        y_raw = points[:, 1]
        if normalized_region is None:
            points[:, 0], points[:, 1] = _scale_cartesian_with_aspect_fit(x_raw, y_raw, referenced)
        else:
            x_bounds, y_bounds = normalized_region
            points[:, 0], points[:, 1] = _scale_cartesian_with_aspect_fit(
                x_raw, y_raw, referenced, x_bounds=x_bounds, y_bounds=y_bounds
            )
    elif view_mode in {"polar_full", "polar_symmetry"}:
        theta = (np.degrees(np.arctan2(points[:, 1], points[:, 0])) + 360.0) % 360.0
        radius = np.linalg.norm(points[:, :2], axis=1)
        x_raw = theta
        y_raw = radius
        if normalized_region is None:
            x_bounds = (0.0, theta_max)
            y_bounds = (float(np.min(radius[referenced])), float(np.max(radius[referenced])))
        else:
            r_bounds, theta_bounds = normalized_region
            x_bounds = theta_bounds
            y_bounds = r_bounds
        points[:, 0] = 1.0 - _scale_values_to_unit_interval(x_raw, *x_bounds)
        points[:, 1] = _scale_values_to_unit_interval(y_raw, *y_bounds)
    if points.shape[1] >= 3:
        points[:, 2] = 0.0
    return points, cell_masks


def _generate_motorlike_cmap(scalars: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Generate discrete color categories and corresponding color list for MotorLikeDiscrete."""
    cats = np.zeros_like(scalars, dtype=int)
    colors = ["white", "red", "gray"]
    cats[scalars >= MATERIAL_ID_THRESHOLD_AIR] = 0
    cats[scalars < MATERIAL_ID_THRESHOLD_AIR] = 0
    cats[scalars < MATERIAL_ID_THRESHOLD_MAGNET] = 1
    cats[scalars < MATERIAL_ID_THRESHOLD_COIL] = 0
    cats[scalars <= MATERIAL_ID_THRESHOLD_CORE] = 2
    cats[scalars < MATERIAL_ID_THRESHOLD_LOW] = 0
    return cats, colors


def _generate_material_color_cmap(
    scalars: np.ndarray, material_color_map: dict[int, str] | None
) -> tuple[np.ndarray, list[str]]:
    """Generate material categories with configured colors overriding motor-like fallback colors."""
    cats, colors = _generate_motorlike_cmap(scalars)
    if not material_color_map:
        return cats, colors
    cats = cats.copy()
    colors = list(colors)
    material_ids = np.asarray(scalars, dtype=int)
    for physical_id, color in material_color_map.items():
        mask = material_ids == int(physical_id)
        if not np.any(mask):
            continue
        cats[mask] = len(colors)
        colors.append(color)
    return cats, colors


def _select_msh_scalar_data(cell_data_vtk: dict[str, np.ndarray], *, scalar_name: str) -> tuple[str, np.ndarray]:
    if scalar_name in cell_data_vtk:
        return scalar_name, np.asarray(cell_data_vtk[scalar_name], dtype=float)
    if "gmsh:physical" not in cell_data_vtk:
        msg = "No material info"
        raise ValueError(msg)
    return "MaterialID", cell_data_vtk["gmsh:physical"]


def convert_msh_to_img(
    msh_filepath: str,
    out_filepath: str,
    cmap: str = "Greys",
    scalar_name: str = "rho_proj",
    material_color_map: dict[int, str] | None = None,
    resolution: int = 480,
    view_mode: str = "cartesian_aspect_fit",
    sym_deg: float = 0.0,
    num_rotate: int = 0,
    image_target_region: tuple[tuple[float, float], tuple[float, float]] | list[list[float]] | None = None,
) -> None:
    """visualize mesh data in msh format, and output as image file

    Args:
        msh_filepath (str): msh file path
        out_filepath (str): output image file path
        cmap (str): pyvista color map. Special keyword "MotorLikeDiscrete" maps material numbers to several color groups
        scalar_name (str): cell data name used when mode is "density"
        material_color_map (dict[int, str] | None): physical ID to color overrides for material images
        resolution (int): square PNG size in pixels
        view_mode (str): cartesian_full, cartesian_aspect_fit, polar_full, or polar_symmetry
        sym_deg (float): symmetry angle in degrees for polar view ranges
        num_rotate (int): rotation copy count for polar_full view range
        image_target_region: optional image region. Cartesian uses ((x1, x2), (y1, y2));
            polar uses ((r1, r2), (theta1, theta2)) with theta in degrees.
    """
    # convert msh for pyvista
    if resolution < 1:
        msg = f"resolution must be >= 1, but got {resolution}"
        raise ValueError(msg)
    mesh = meshio.read(msh_filepath)
    points, cell_masks = _prepare_image_points_and_masks(
        mesh, view_mode=view_mode, sym_deg=sym_deg, num_rotate=num_rotate, image_target_region=image_target_region
    )
    cells_vtk, celltypes_vtk, cell_data_vtk = _convert_msh_to_vtk_celldata(mesh, cell_masks=cell_masks)
    ugrid = pv.UnstructuredGrid(cells_vtk, celltypes_vtk, points)
    scalar_key, scalar_values = _select_msh_scalar_data(cell_data_vtk, scalar_name=scalar_name)
    ugrid.cell_data[scalar_key] = scalar_values
    # create img
    plotter = pv.Plotter(off_screen=True)
    if scalar_key == "MaterialID" and cmap == "MotorLikeDiscrete":
        cats, colors = _generate_material_color_cmap(
            scalars=np.array(scalar_values), material_color_map=material_color_map
        )
        ugrid.cell_data["cats"] = cats
        plotter.add_mesh(
            ugrid, scalars="cats", cmap=colors, clim=(0, len(colors) - 1), show_edges=False, show_scalar_bar=False
        )
    else:
        kwargs = {}
        if scalar_key == scalar_name:
            kwargs["clim"] = (0.0, 1.0)
        plotter.add_mesh(ugrid, scalars=scalar_key, cmap="Greys", show_edges=False, show_scalar_bar=False, **kwargs)
    plotter.view_xy()
    plotter.camera.parallel_projection = True
    plotter.camera_position = [(0.5, 0.5, 1.0), (0.5, 0.5, 0.0), (0.0, 1.0, 0.0)]
    plotter.camera.parallel_scale = 0.5
    plotter.screenshot(out_filepath, window_size=[resolution, resolution])
    plotter.close()
    plotter.clear()


def output_outcome(
    individual: Individual,
    filepath: str,
    material_color_map: dict[int, str] | None = None,
    image_export: dict[str, object] | None = None,
) -> str | None:
    """output outcome of individual

    Args:
        filepath (str): filepath for outcome data.

    Returns:
        str: filepath of copied outcome
    """
    if individual.outcome_filepath is not None:
        individual_filepath = Path(individual.outcome_filepath)
        _filepath = Path(filepath)
        dst_filepath = str(_filepath.with_suffix(individual_filepath.suffix))
        with contextlib.suppress(shutil.SameFileError):  # trial of copying same file is allowed
            shutil.copy(individual_filepath, dst_filepath)
        # convert msh to png image
        try:
            image_export_kwargs = dict(image_export or {})
            convert_msh_to_img(
                str(_filepath.with_suffix(".msh")),
                str(Path(dst_filepath).with_suffix(".png")),
                cmap="MotorLikeDiscrete",
                material_color_map=material_color_map,
                **image_export_kwargs,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("Failed to convert %s to PNG: %s. Continue...", individual_filepath, exc)
    else:
        dst_filepath = None
    return dst_filepath
