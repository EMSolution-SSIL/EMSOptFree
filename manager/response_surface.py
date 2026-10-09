"""
response_surface.py
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

import re
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from core.optimizer.surrogate_models.random_forest import RandomForestSurrogate
from emsopt_engine.configs.optimization import ResponseSurfaceConfig

logger = getLogger(__name__)

MIN_CONTOUR_DESIGN_COLUMNS = 2
ANCHOR_ROUND_DIGITS = 8


@dataclass(slots=True)
class ResponseSurfaceRequest:
    """Manager-side response surface generation request sent from the GUI."""

    request_id: str
    target_column: str
    x_column: str
    y_column: str
    anchor_solution: list[float] | None = None
    anchor_source: Literal["selected_individual", "median_record"] = "median_record"
    grid_size: int | None = None


@dataclass(slots=True)
class ResponseSurfaceSession:
    """Lazy response surface generator backed by run records and RF caches."""

    summary_dir: Path
    source_records: Path
    config: ResponseSurfaceConfig
    numeric_df: pd.DataFrame
    design_columns: list[str]
    target_columns: list[str]
    _models: dict[str, RandomForestSurrogate] = field(default_factory=dict)
    _importance_cache: dict[str, dict] = field(default_factory=dict)
    _surface_cache: dict[tuple, dict] = field(default_factory=dict)

    def build_manifest(self) -> dict:
        """Return lightweight metadata needed by the GUI before any plot is generated."""
        return {
            "schema_version": 2,
            "source_records": str(self.source_records),
            "surrogate_model": self.config.surrogate_model,
            "num_samples": len(self.numeric_df),
            "grid_size": self.config.grid_size,
            "design_columns": list(self.design_columns),
            "target_columns": list(self.target_columns),
            "default_anchor": self.default_anchor.tolist(),
        }

    @property
    def default_anchor(self) -> np.ndarray:
        """Use the median design point as a stable fallback anchor."""
        return self.numeric_df[self.design_columns].median(axis=0).to_numpy(dtype=float)

    def create_response(self, request: ResponseSurfaceRequest) -> dict:
        """Resolve one GUI request into contour-grid data and feature importance data."""
        if request.target_column not in self.target_columns:
            msg = f"Unknown response surface target: {request.target_column}"
            raise ValueError(msg)
        if request.x_column not in self.design_columns or request.y_column not in self.design_columns:
            msg = f"Unknown response surface axis selection: x={request.x_column}, y={request.y_column}"
            raise ValueError(msg)
        if request.x_column == request.y_column:
            msg = "Response surface axes must be different."
            raise ValueError(msg)

        anchor, anchor_source = self._resolve_anchor(request.anchor_solution, request.anchor_source)
        grid_size = request.grid_size or self.config.grid_size
        model = self._get_model(request.target_column)
        contour = self._get_surface(
            target_column=request.target_column,
            x_column=request.x_column,
            y_column=request.y_column,
            anchor=anchor,
            grid_size=grid_size,
            model=model,
        )
        importance = self._get_importance(request.target_column, model)
        return {
            "request_id": request.request_id,
            "target": request.target_column,
            "x": request.x_column,
            "y": request.y_column,
            "anchor_source": anchor_source,
            "anchor_solution": anchor.tolist(),
            "contour": contour,
            "importance": importance,
        }

    def _resolve_anchor(
        self,
        anchor_solution: list[float] | None,
        anchor_source: Literal["selected_individual", "median_record"],
    ) -> tuple[np.ndarray, Literal["selected_individual", "median_record"]]:
        """Clip the incoming anchor to the design-variable dimension, or fall back to the median."""
        if anchor_solution is None:
            return self.default_anchor.copy(), "median_record"
        anchor = np.asarray(anchor_solution, dtype=float).reshape(-1)
        if anchor.size < len(self.design_columns):
            logger.warning(
                "Response surface anchor size mismatch. Expected at least %d, got %d. Use default anchor.",
                len(self.design_columns),
                anchor.size,
            )
            return self.default_anchor.copy(), "median_record"
        return anchor[: len(self.design_columns)].copy(), anchor_source

    def _get_model(self, target_column: str) -> RandomForestSurrogate:
        """Train one RF per target column and reuse it across GUI requests."""
        model = self._models.get(target_column)
        if model is not None:
            return model
        model = RandomForestSurrogate(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=self.config.random_state,
        )
        model.fit(
            self.numeric_df[self.design_columns].to_numpy(dtype=float),
            self.numeric_df[target_column].to_numpy(dtype=float),
        )
        self._models[target_column] = model
        return model

    def _get_importance(self, target_column: str, model: RandomForestSurrogate) -> dict:
        """Cache normalized RF feature importances because they depend only on the target."""
        cached = self._importance_cache.get(target_column)
        if cached is not None:
            return cached
        importances = model.get_feature_importances()
        total = float(importances.sum())
        ratios = importances / total if total > 0.0 else np.zeros_like(importances)
        cached = {
            "feature_names": [str(i + 1) for i in range(len(self.design_columns))],
            "values": ratios.astype(float).tolist(),
        }
        self._importance_cache[target_column] = cached
        return cached

    def _get_surface(
        self,
        *,
        target_column: str,
        x_column: str,
        y_column: str,
        anchor: np.ndarray,
        grid_size: int,
        model: RandomForestSurrogate,
    ) -> dict:
        """Generate a 2-axis response map while fixing the remaining variables at the anchor."""
        key = (
            target_column,
            x_column,
            y_column,
            grid_size,
            tuple(round(float(v), ANCHOR_ROUND_DIGITS) for v in anchor),
        )
        # Cache by target, axis pair, grid size, and rounded anchor to avoid noisy re-generation.
        cached = self._surface_cache.get(key)
        if cached is not None:
            return cached

        x_values = np.linspace(
            float(self.numeric_df[x_column].min()),
            float(self.numeric_df[x_column].max()),
            grid_size,
        )
        y_values = np.linspace(
            float(self.numeric_df[y_column].min()),
            float(self.numeric_df[y_column].max()),
            grid_size,
        )
        xx, yy = np.meshgrid(x_values, y_values)
        grid = np.tile(anchor, (xx.size, 1))
        x_idx = self.design_columns.index(x_column)
        y_idx = self.design_columns.index(y_column)
        grid[:, x_idx] = xx.ravel()
        grid[:, y_idx] = yy.ravel()
        pred, _ = model.predict(grid)
        zz = pred[:, 0].reshape(xx.shape)
        cached = {
            "rows": int(zz.shape[0]),
            "cols": int(zz.shape[1]),
            "z_values": zz.astype(float).ravel().tolist(),
            "bounds_min": [float(x_values.min()), float(y_values.min())],
            "bounds_max": [float(x_values.max()), float(y_values.max())],
            "scale_min": float(zz.min()),
            "scale_max": float(zz.max()),
            "sample_points": {
                "x": self.numeric_df[x_column].astype(float).tolist(),
                "y": self.numeric_df[y_column].astype(float).tolist(),
            },
        }
        self._surface_cache[key] = cached
        return cached


def build_response_surface_session(
    summary_dir: str | Path,
    config: ResponseSurfaceConfig,
) -> ResponseSurfaceSession | None:
    """Load records and build a lazy response-surface session for check-time GUI use."""
    summary_path = Path(summary_dir)
    records_csv = summary_path / "records" / "run_records.csv"
    if not records_csv.exists():
        logger.warning("Run records are missing at %s. Skip response surface generation.", records_csv)
        return None

    df = pd.read_csv(records_csv)
    df = _filter_success_records(df)
    design_columns = _resolve_columns(df, config.design_columns, prefix="solution_")
    target_columns = _resolve_target_columns(df, config.target_columns)
    if len(design_columns) < MIN_CONTOUR_DESIGN_COLUMNS:
        logger.warning("At least two design variable columns are required for response surface contours.")
        return None
    if not target_columns:
        logger.warning("No objective or label columns are available for response surface targets.")
        return None

    numeric_df = _numeric_training_frame(df, [*design_columns, *target_columns])
    if len(numeric_df) < config.min_samples:
        logger.warning(
            "Only %d valid records are available for response surface generation. Required at least %d.",
            len(numeric_df),
            config.min_samples,
        )
        return None
    return ResponseSurfaceSession(
        summary_dir=summary_path,
        source_records=records_csv,
        config=config,
        numeric_df=numeric_df,
        design_columns=design_columns,
        target_columns=target_columns,
    )


def _filter_success_records(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only successful evaluations when status is available."""
    if "record_status" not in df.columns:
        return df
    return df[df["record_status"].fillna("success").astype(str) == "success"].copy()


def _resolve_columns(df: pd.DataFrame, configured_columns: list[str] | None, *, prefix: str) -> list[str]:
    """Resolve design-variable columns from config or `solution_*` defaults."""
    if configured_columns is not None:
        missing = [column for column in configured_columns if column not in df.columns]
        if missing:
            msg = f"Configured response surface columns are missing: {missing}"
            raise ValueError(msg)
        return list(configured_columns)
    return _sort_numbered_columns([column for column in df.columns if column.startswith(prefix)])


def _resolve_target_columns(df: pd.DataFrame, configured_columns: list[str] | None) -> list[str]:
    """Resolve target columns from config or ``label__*` defaults."""
    if configured_columns is not None:
        missing = [column for column in configured_columns if column not in df.columns]
        if missing:
            msg = f"Configured response surface target columns are missing: {missing}"
            raise ValueError(msg)
        return list(configured_columns)
    return sorted([column for column in df.columns if column.startswith("label__")])


def _sort_numbered_columns(columns: list[str]) -> list[str]:
    """Sort numbered columns like `solution_1`, `solution_2`, ... in natural order."""

    def _key(column: str) -> tuple[str, int]:
        match = re.search(r"_(\d+)$", column)
        return re.sub(r"_\d+$", "", column), int(match.group(1)) if match else 0

    return sorted(columns, key=_key)


def _numeric_training_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Coerce training columns to numeric and drop rows that cannot be learned safely."""
    numeric_df = df[columns].apply(pd.to_numeric, errors="coerce")
    return numeric_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
