"""
gradient_update.py
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

from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import Literal

import numpy as np

from emsopt_engine.configs.base import BaseConfig
from emsopt_engine.individual import Individual, Population
from emsopt_engine.interface.optimizer_interface import OptimizerInterface
from emsopt_engine.interface.so_optimizer_base import SOOptimizerBase
from emsopt_engine.registry import optimizer
from utils.utils import read_variable_length_float_csv, read_variable_length_int_csv

logger = getLogger(__name__)


class GradientUpdate(SOOptimizerBase, OptimizerInterface):
    EPS_NORMALIZER = 1e-30

    def __init__(  # noqa: PLR0913
        self,
        dim: int,
        init_uniform_value: float = 0.0,
        init_value_filepath: str | None = None,
        bounds: tuple[float, float] | None = None,
        step_size: float = 0.1,
        reaction_coef: float = 1.0,
        diffusion_coef: float = 1.0,
        damping_factor: float = 0.9,
        scaling_mode: Literal["none", "max", "sign"] = "max",
        constraint_mode: Literal["none", "volume_penalty"] = "none",
        volume_frac_ulim: float = 0.5,
        volume_penalty_coef: float = 1e3,
    ) -> None:
        super().__init__(1, None)  # Call SOOptimizerBase's __init__ according to Python's MRO
        if not isinstance(dim, int) or dim <= 0:
            msg = "dim must be a positive integer"
            raise ValueError(msg)
        self.dim = dim
        self.bounds = bounds if bounds is not None else (-1, 1)
        self.init_uniform_value = init_uniform_value
        self.init_value_filepath = init_value_filepath
        self.step_size = step_size
        self.reaction_coef = reaction_coef
        self.diffusion_coef = diffusion_coef
        self.damping_factor = damping_factor
        self.scaling_mode = scaling_mode
        self.constraint_mode = constraint_mode
        self.volume_frac_ulim = volume_frac_ulim
        self.volume_penalty_coef = volume_penalty_coef
        if not isinstance(damping_factor, float) or damping_factor > 1.0:
            msg = "damping_factor must be <= 1.0."
            raise ValueError(msg)
        self.old_obj_value = 1e10

    def setup_population(self, evaluated_population: Population) -> None:  # noqa: ARG002 ; for interface compatibility
        """Not needed"""
        return

    def _generate_init_solution(self) -> np.ndarray:
        if self.init_value_filepath is not None:
            return np.loadtxt(self.init_value_filepath, delimiter=",")
        return np.clip(np.full(self.dim, float(self.init_uniform_value)), self.bounds[0], self.bounds[1])

    def get_candidates(self) -> Population:
        """just return current solution"""
        if self.population is None:
            init_solution = self._generate_init_solution()
            self.population = Population({0: Individual(solution=list(init_solution))})
        return self.population

    def proceed_to_next_iteration(self, evaluated_candidates: Population) -> None:
        # load necessary informations
        case_name = "transient"
        phi = np.loadtxt(
            Path(evaluated_candidates[0].working_dir) / case_name / BaseConfig.DESIGN_LS_PARAMETERS_FILENAME.value,
            delimiter=",",
        )  # load current value considering when IDM is enabled
        self.dim = len(phi)
        gradient = np.loadtxt(
            Path(evaluated_candidates[0].working_dir) / case_name / BaseConfig.DESIGN_GRADIENT_FILENAME.value
        )
        volumes = np.loadtxt(
            Path(evaluated_candidates[0].working_dir) / case_name / BaseConfig.DESIGN_VOLUMES_FILENAME.value
        )
        material_density = np.loadtxt(
            Path(evaluated_candidates[0].working_dir) / case_name / BaseConfig.DESIGN_LS_OUTPUT_FILENAME.value
        )
        neighbors = read_variable_length_int_csv(
            Path(evaluated_candidates[0].working_dir) / case_name / BaseConfig.DESIGN_NEIGHBORS_FILENAME.value
        )
        neighbor_distances_path = (
            Path(evaluated_candidates[0].working_dir) / case_name / BaseConfig.DESIGN_NEIGHBOR_DISTANCES_FILENAME.value
        )
        if neighbor_distances_path.is_file():
            neighbor_distances = read_variable_length_float_csv(neighbor_distances_path)
        else:
            neighbor_distances = None
            logger.warning(
                "Neighbor distance CSV does not exist; using legacy equal-weight graph calculation: path=%s",
                neighbor_distances_path,
            )
        gradient = compute_graph_smoothing(gradient, neighbors, neighbor_distances)
        gradient = self.scale_arr(gradient)
        laplacian = compute_graph_laplacian(material_density, neighbors, neighbor_distances)
        total_volume = np.sum(volumes)
        iron_volume = np.dot(material_density, volumes)
        volumes_fraction = volumes / total_volume
        # Shrink the step size when the objective worsens.
        current_obj_value = evaluated_candidates[0].metrics.fitness
        if current_obj_value > self.old_obj_value:
            self.step_size *= self.damping_factor
            logger.info("Decreased step_size to: %.8f", self.step_size)
        self.old_obj_value = evaluated_candidates[0].metrics.fitness
        # Phase-field reaction-diffusion equation: dphi/dt = -M * dF/dphi + tau * Laplacian(phi).
        reaction = -self.reaction_coef * gradient
        diffusion = self.diffusion_coef * laplacian
        if self.constraint_mode == "volume_penalty":
            violation = max(0.0, iron_volume / total_volume - self.volume_frac_ulim)
            reaction -= self.volume_penalty_coef * violation * volumes_fraction
            evaluated_candidates[0].metrics.ineq_constraints.append(violation)  # just for recording
        # Update rule: phi += dphi = dphi/dt * dt
        move_amount = self.step_size * (reaction + diffusion)
        evaluated_candidates[0].solution = list(np.clip(phi + move_amount, self.bounds[0], self.bounds[1]))
        self.population = deepcopy(evaluated_candidates)
        self.update_best_individual()

    def scale_arr(self, arr: np.ndarray) -> np.ndarray:
        arr_cp = np.array(arr, dtype=np.float64)
        nonzero_arr = arr_cp[np.abs(arr_cp) > self.EPS_NORMALIZER]
        if nonzero_arr.size == 0:
            logger.warning("Array values are all zeros. Skip normalization.")
            return arr_cp
        if self.scaling_mode == "none":
            return arr_cp
        if self.scaling_mode == "max":
            normalizer = float(np.max(np.abs(nonzero_arr)))
            return arr_cp * (self.bounds[1] - self.bounds[0]) / normalizer
        if self.scaling_mode == "sign":
            arr_cp[arr_cp < 0] = self.bounds[0]
            arr_cp[arr_cp > 0] = self.bounds[1]
            return arr_cp
        logger.warning("Unknown scaling mode: %s. Skip normalization.", self.scaling_mode)
        return arr_cp


def _validate_neighbor_distances(
    neighbors: list[np.ndarray], neighbor_distances: list[np.ndarray], num_values: int
) -> None:
    if len(neighbors) != num_values or len(neighbor_distances) != num_values:
        msg = (
            "Neighbor graph size must match values: "
            f"values={num_values}, neighbors={len(neighbors)}, distances={len(neighbor_distances)}"
        )
        raise ValueError(msg)
    for index, (neighbor_ids, distances) in enumerate(zip(neighbors, neighbor_distances, strict=True)):
        if len(neighbor_ids) != len(distances):
            msg = (
                "Neighbor IDs and distances must have matching lengths: "
                f"index={index}, neighbors={len(neighbor_ids)}, distances={len(distances)}"
            )
            raise ValueError(msg)
        if np.any(np.asarray(distances) < 0.0):
            msg = f"Neighbor distances must be non-negative: index={index}"
            raise ValueError(msg)


def compute_graph_smoothing(
    values: np.ndarray,
    neighbors: list[np.ndarray],
    neighbor_distances: list[np.ndarray] | None = None,
) -> np.ndarray:
    n = len(values)
    if neighbor_distances is None:
        if len(neighbors) != n:
            return np.zeros_like(values)
        res = np.zeros_like(values)
        for i in range(n):
            nb = neighbors[i]
            if nb.size > 0:
                res[i] = (np.sum(values[nb]) + values[i]) / (len(nb) + 1)
        return res

    _validate_neighbor_distances(neighbors, neighbor_distances, n)

    res = np.zeros_like(values)
    for i in range(n):
        nb = neighbors[i]
        if nb.size == 0:
            continue
        distances = neighbor_distances[i]
        distance_sum = float(np.sum(distances))
        if distance_sum <= 0.0:
            continue
        res[i] = float(np.dot(distances, values[nb]) / distance_sum)
    return res


def compute_graph_laplacian(
    values: np.ndarray,
    neighbors: list[np.ndarray],
    neighbor_distances: list[np.ndarray] | None = None,
) -> np.ndarray:
    n = len(values)
    if neighbor_distances is None:
        if len(neighbors) != n:
            return np.zeros_like(values)
        res = np.zeros_like(values)
        for i in range(n):
            nb = neighbors[i]
            if nb.size > 0:
                res[i] = float(np.mean(values[nb] - values[i]))
        return res

    _validate_neighbor_distances(neighbors, neighbor_distances, n)

    res = np.zeros_like(values)
    for i in range(n):
        nb = neighbors[i]
        if nb.size == 0:
            continue
        distances = neighbor_distances[i]
        distance_sum = float(np.sum(distances))
        if distance_sum <= 0.0:
            continue
        res[i] = float(np.dot(distances, values[nb] - values[i]) / distance_sum)
    return res


@optimizer("gradient_update")
def build_gradient_update(  # noqa: PLR0913
    dim: int,
    num_obj: int,  # noqa: ARG001 ; for interface compatibility
    init_uniform_value: float = 0.0,
    init_value_filepath: str | None = None,
    bounds: tuple[float, float] | None = None,
    step_size: float = 0.1,
    reaction_coef: float = 1.0,
    diffusion_coef: float = 1.0,
    damping_factor: float = 0.9,
    scaling_mode: Literal["none", "max", "sign"] = "max",
    constraint_mode: Literal["none", "volume_penalty"] = "none",
    volume_frac_ulim: float = 0.5,
    volume_penalty_coef: float = 1e3,
) -> OptimizerInterface:
    return GradientUpdate(
        dim,
        init_uniform_value,
        init_value_filepath,
        bounds,
        step_size,
        reaction_coef,
        diffusion_coef,
        damping_factor,
        scaling_mode,
        constraint_mode,
        volume_frac_ulim,
        volume_penalty_coef,
    )
