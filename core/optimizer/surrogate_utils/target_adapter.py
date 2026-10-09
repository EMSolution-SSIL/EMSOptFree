"""
target_adapter.py
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

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

from emsopt_engine.individual import Individual, OptimizationProblemMetrics


class SurrogateTargetAdapter:
    """Resolve surrogate training/prediction targets for an optimizer.

    When `func_manager` is absent, the adapter falls back to the legacy
    objective-direct mode. When it is present, raw `label_values` become the
    surrogate's target space and predicted labels are converted back into
    `OptimizationProblemMetrics`.
    """

    def __init__(self, func_manager: object | None = None) -> None:
        if func_manager is not None and not hasattr(func_manager, "get_label_names"):
            msg = "func_manager must provide get_label_names() when used for surrogate targets"
            raise TypeError(msg)
        if func_manager is not None and not hasattr(func_manager, "compute_all_from_label_values"):
            msg = "func_manager must provide compute_all_from_label_values() when used for surrogate targets"
            raise TypeError(msg)
        self.func_manager = func_manager
        self.target_label_names = list(func_manager.get_label_names()) if func_manager is not None else []

    @property
    def uses_label_targets(self) -> bool:
        """Return whether surrogate targets are label-based instead of objective-based."""
        return len(self.target_label_names) > 0

    @property
    def num_targets(self) -> int | None:
        """Return the surrogate output dimension when label-based targets are active."""
        if not self.uses_label_targets:
            return None
        return len(self.target_label_names)

    def extract_training_targets(self, individual: Individual) -> np.ndarray:
        """Build the `y` vector used for surrogate training from one individual."""
        if not self.uses_label_targets:
            return np.asarray(individual.metrics.objectives, dtype=float)
        missing = [label for label in self.target_label_names if label not in individual.label_values]
        if missing:
            msg = "label_values are missing required surrogate labels: " + ", ".join(missing)
            raise ValueError(msg)
        return np.asarray([individual.label_values[label] for label in self.target_label_names], dtype=float)

    def build_predicted_label_values(self, predicted_targets: np.ndarray | list[float]) -> dict[str, float]:
        """Map surrogate outputs back to the fixed label order resolved at setup time."""
        if not self.uses_label_targets:
            return {}
        values = np.asarray(predicted_targets, dtype=float).reshape(-1).tolist()
        if len(values) != len(self.target_label_names):
            msg = (
                "surrogate prediction dimension does not match required label count: "
                f"expected {len(self.target_label_names)}, got {len(values)}"
            )
            raise ValueError(msg)
        return {label: float(value) for label, value in zip(self.target_label_names, values, strict=True)}

    def apply_prediction(
        self, individual: Individual, predicted_targets: np.ndarray | list[float]
    ) -> tuple[list[float], dict[str, float] | None]:
        """Apply surrogate outputs to an individual and return objective-space values.

        Returns a tuple of:
        - objective values used by the optimizer for ranking/selection
        - raw predicted labels when label mode is active
        """
        if not self.uses_label_targets:
            objectives = np.asarray(predicted_targets, dtype=float).reshape(-1).tolist()
            individual.metrics.objectives = objectives
            return objectives, None
        predicted_label_values = self.build_predicted_label_values(predicted_targets)
        individual.label_values = {**individual.label_values, **predicted_label_values}
        # Reconstruct transformed objectives/constraints from predicted raw labels.
        metrics = self.func_manager.compute_all_from_label_values(individual.label_values)
        if not isinstance(metrics, OptimizationProblemMetrics):
            msg = f"func_manager must return OptimizationProblemMetrics, got {type(metrics)}"
            raise TypeError(msg)
        individual.metrics = metrics
        return list(metrics.objectives), predicted_label_values

    def true_label_values_for_tracking(self, individual: Individual) -> Mapping[str, float] | None:
        """Return label_values to be copied into surrogate tracking, when relevant."""
        if not self.uses_label_targets or not individual.label_values:
            return None
        return dict(individual.label_values)
