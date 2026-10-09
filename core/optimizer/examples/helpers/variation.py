"""
variation.py
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
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import numpy as np

from emsopt_engine.individual import Individual, Population

DEFAULT_BOUNDS = (-1, 1)
DEFAULT_EPS = 1e-10
HALF_PROBABILITY = 0.5
VariableTypeCounts = dict[Literal["categorical", "discrete", "continuous"], int]


def normalize_mean(mean: np.ndarray | None, dim: int) -> np.ndarray | None:
    """Pad a short mean vector with zeros and reject vectors longer than dim."""
    if mean is None:
        return None

    normalized = np.asarray(mean, dtype=float)
    if normalized.ndim != 1:
        msg = "mean must be a one-dimensional vector"
        raise ValueError(msg)
    if len(normalized) > dim:
        msg = f"mean length ({len(normalized)}) must not exceed dim ({dim})"
        raise ValueError(msg)
    if len(normalized) < dim:
        normalized = np.pad(normalized, (0, dim - len(normalized)), constant_values=0.0)
    return normalized


@dataclass(frozen=True)
class VariableLayout:
    """Contiguous mixed-variable layout in Individual.solution."""

    num_categorical: int = 0
    num_discrete: int = 0
    num_continuous: int = 0

    @property
    def dim(self) -> int:
        return self.num_categorical + self.num_discrete + self.num_continuous

    @property
    def categorical_slice(self) -> slice:
        return slice(0, self.num_categorical)

    @property
    def discrete_slice(self) -> slice:
        start = self.num_categorical
        return slice(start, start + self.num_discrete)

    @property
    def continuous_slice(self) -> slice:
        start = self.num_categorical + self.num_discrete
        return slice(start, start + self.num_continuous)

    @property
    def is_continuous_only(self) -> bool:
        return self.num_categorical == 0 and self.num_discrete == 0


def normalize_variable_layout(
    variable_type_counts: VariableTypeCounts | None,
    dim: int,
) -> VariableLayout:
    """
    Normalize mixed-variable counts to a contiguous layout.

    If omitted, all variables are treated as continuous for backward compatibility.
    When continuous is omitted, it is inferred from dim after categorical/discrete counts.
    """
    if variable_type_counts is None:
        return VariableLayout(num_continuous=dim)

    unknown_keys = set(variable_type_counts) - {"categorical", "discrete", "continuous"}
    if unknown_keys:
        msg = f"unknown variable type keys: {sorted(unknown_keys)}"
        raise ValueError(msg)

    num_categorical = variable_type_counts.get("categorical", 0)
    num_discrete = variable_type_counts.get("discrete", 0)
    num_continuous = variable_type_counts.get("continuous")
    values = [num_categorical, num_discrete]
    if num_continuous is not None:
        values.append(num_continuous)
    if not all(isinstance(value, int) and value >= 0 for value in values):
        msg = "variable type counts must be non-negative integers"
        raise ValueError(msg)

    if num_continuous is None:
        num_continuous = dim - num_categorical - num_discrete
    if num_continuous < 0:
        msg = "sum of variable type counts must match dim"
        raise ValueError(msg)
    layout = VariableLayout(
        num_categorical=num_categorical,
        num_discrete=num_discrete,
        num_continuous=num_continuous,
    )
    if layout.dim != dim:
        msg = "sum of variable type counts must match dim"
        raise ValueError(msg)
    return layout


def normalize_choice_values(
    values: Sequence[Sequence[float | int]] | None,
    expected_length: int,
    name: str,
) -> list[list[float | int]]:
    """Validate numeric choice lists for categorical or discrete variables."""
    if expected_length == 0:
        return []
    if values is None:
        msg = f"{name} must be provided when the corresponding variable count is positive"
        raise ValueError(msg)
    if len(values) != expected_length:
        msg = f"{name} length must match the corresponding variable count"
        raise ValueError(msg)

    normalized: list[list[float | int]] = []
    for idx, choices in enumerate(values):
        if not choices:
            msg = f"{name}[{idx}] must not be empty"
            raise ValueError(msg)
        if not all(isinstance(choice, int | float | np.integer | np.floating) for choice in choices):
            msg = f"{name}[{idx}] must contain only numeric values"
            raise ValueError(msg)
        normalized.append([choice.item() if isinstance(choice, np.generic) else choice for choice in choices])
    return normalized


def _nearest_choice(value: float, choices: Sequence[float | int]) -> float | int:
    return min(choices, key=lambda choice: abs(float(value) - float(choice)))


def repair_mixed_solution(
    ind: Individual,
    layout: VariableLayout,
    categorical_choices: list[list[float | int]],
    discrete_values: list[list[float | int]],
    bounds: list[tuple[float, float]],
) -> Individual:
    """Repair an individual so each variable belongs to its configured domain."""
    repaired = deepcopy(ind)
    for offset, choices in enumerate(categorical_choices):
        value = repaired.solution[offset]
        if value not in choices:
            msg = f"categorical variable {offset} value {value} is not in allowed choices"
            raise ValueError(msg)

    discrete_start = layout.discrete_slice.start
    for offset, choices in enumerate(discrete_values):
        idx = discrete_start + offset
        repaired.solution[idx] = _nearest_choice(repaired.solution[idx], choices)

    for idx in range(layout.continuous_slice.start, layout.continuous_slice.stop):
        low, high = bounds[idx - layout.continuous_slice.start]
        repaired.solution[idx] = np.clip(repaired.solution[idx], low, high)
    return repaired


def normalize_bounds(
    bounds: tuple[float, float] | list[tuple[float, float]] | None,
    dim: int,
) -> list[tuple[float, float]]:
    """
    Normalize optimizer bounds to one (low, high) tuple per variable.

    Args:
        bounds (tuple[float, float] | list[tuple[float, float]] | None): Bounds configuration.
        dim (int): Optimization problem dimension.

    Returns:
        list[tuple[float, float]]: Bounds for each variable.

    Raises:
        ValueError: If bounds shape is invalid.
    """
    if bounds is None:
        return [DEFAULT_BOUNDS] * dim
    bounds_array = np.asarray(bounds)
    if bounds_array.ndim == 1:
        return [tuple(bounds)] * dim
    if bounds_array.ndim == 2 and len(bounds) <= dim:  # noqa: PLR2004
        normalized_bounds = list(bounds)
        normalized_bounds.extend([DEFAULT_BOUNDS] * (dim - len(bounds)))
        return normalized_bounds
    msg = "bounds must be a tuple or list of tuples with length equal to or smaller than dim."
    raise ValueError(msg)


def sbx_crossover(
    parent1: Individual,
    parent2: Individual,
    bounds: list[tuple[float, float]],
    rng: np.random.Generator,
    eta: float = 20,
    pc: float = 0.75,
    eps: float = DEFAULT_EPS,
) -> tuple[Individual, Individual]:
    """
    Perform Simulated Binary Crossover (SBX) between two parents.

    Args:
        parent1 (Individual): The first parent individual.
        parent2 (Individual): The second parent individual.
        bounds (list[tuple[float, float]]): Variable bounds.
        rng (np.random.Generator): Random number generator.
        eta (float): Distribution index for SBX.
        pc (float): Crossover probability per variable.
        eps (float): Threshold for treating parent values as equal.

    Returns:
        tuple[Individual, Individual]: Two offspring individuals.
    """
    child1, child2 = deepcopy(parent1), deepcopy(parent2)
    for i, (x_low, x_up) in enumerate(bounds):
        if rng.random() > pc:
            continue
        x1, x2 = parent1.solution[i], parent2.solution[i]
        if x_up - x_low <= 0:
            continue
        if abs(x1 - x2) > eps:
            u = rng.random()
            if u < HALF_PROBABILITY:
                beta = (2.0 * u) ** (1.0 / (eta + 1.0))
            else:
                beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
            if x1 > x2:
                x1, x2 = x2, x1
            child1.solution[i] = np.clip(0.5 * ((x1 + x2) - beta * abs(x2 - x1)), x_low, x_up)
            child2.solution[i] = np.clip(0.5 * ((x1 + x2) + beta * abs(x2 - x1)), x_low, x_up)
    return child1, child2


def polynomial_mutation(
    ind: Individual,
    bounds: list[tuple[float, float]],
    rng: np.random.Generator,
    eta: float = 20,
    pm: float | None = None,
) -> Individual:
    """
    Apply polynomial mutation to an individual.

    Args:
        ind (Individual): The individual to mutate.
        bounds (list[tuple[float, float]]): Variable bounds.
        rng (np.random.Generator): Random number generator.
        eta (float): Distribution index for mutation.
        pm (float | None): Mutation probability. Defaults to 1/dim.

    Returns:
        Individual: The mutated individual.
    """
    mutated_ind = deepcopy(ind)
    dim = len(bounds)
    if pm is None:
        pm = 1.0 / dim
    for i, (x_low, x_up) in enumerate(bounds):
        if x_up - x_low <= 0:
            continue
        if rng.random() > pm:
            continue
        delta1 = (ind.solution[i] - x_low) / (x_up - x_low)
        delta2 = (x_up - ind.solution[i]) / (x_up - x_low)
        u = rng.random()
        if u < HALF_PROBABILITY:
            val = 2.0 * u + (1.0 - 2.0 * u) * ((1 - delta1) ** (eta + 1))
            delta = val ** (1.0 / (eta + 1.0)) - 1.0
        else:
            val = 2.0 * (1.0 - u) + 2.0 * (u - HALF_PROBABILITY) * ((1 - delta2) ** (eta + 1))
            delta = 1.0 - val ** (1.0 / (eta + 1.0))
        mutated_ind.solution[i] = np.clip(ind.solution[i] + delta * (x_up - x_low), x_low, x_up)
    return mutated_ind


def mixed_crossover(
    parent1: Individual,
    parent2: Individual,
    layout: VariableLayout,
    categorical_choices: list[list[float | int]],
    discrete_values: list[list[float | int]],
    bounds: list[tuple[float, float]],
    rng: np.random.Generator,
    eta: float = 20,
    pc: float = 0.75,
    eps: float = DEFAULT_EPS,
) -> tuple[Individual, Individual]:
    """
    Perform mixed-variable crossover.

    Categorical variables use uniform inheritance. Discrete and continuous variables
    use SBX, then discrete variables are repaired to their nearest allowed value.
    """
    child1, child2 = deepcopy(parent1), deepcopy(parent2)
    for idx in range(layout.categorical_slice.start, layout.categorical_slice.stop):
        if rng.random() <= pc and rng.random() < HALF_PROBABILITY:
            child1.solution[idx], child2.solution[idx] = parent2.solution[idx], parent1.solution[idx]

    for idx in range(layout.discrete_slice.start, layout.continuous_slice.stop):
        if rng.random() > pc:
            continue
        if idx < layout.continuous_slice.start:
            values = discrete_values[idx - layout.discrete_slice.start]
            x_low, x_up = min(values), max(values)
        else:
            x_low, x_up = bounds[idx - layout.continuous_slice.start]
        if x_up - x_low <= 0:
            continue
        x1, x2 = parent1.solution[idx], parent2.solution[idx]
        if abs(x1 - x2) <= eps:
            continue
        u = rng.random()
        if u < HALF_PROBABILITY:
            beta = (2.0 * u) ** (1.0 / (eta + 1.0))
        else:
            beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
        if x1 > x2:
            x1, x2 = x2, x1
        child1.solution[idx] = np.clip(0.5 * ((x1 + x2) - beta * abs(x2 - x1)), x_low, x_up)
        child2.solution[idx] = np.clip(0.5 * ((x1 + x2) + beta * abs(x2 - x1)), x_low, x_up)

    return (
        repair_mixed_solution(child1, layout, categorical_choices, discrete_values, bounds),
        repair_mixed_solution(child2, layout, categorical_choices, discrete_values, bounds),
    )


def mixed_mutation(
    ind: Individual,
    layout: VariableLayout,
    categorical_choices: list[list[float | int]],
    discrete_values: list[list[float | int]],
    bounds: list[tuple[float, float]],
    rng: np.random.Generator,
    eta: float = 20,
    pm: float | None = None,
) -> Individual:
    """
    Apply mixed-variable mutation.

    Categorical variables are resampled from allowed choices. Discrete variables
    use polynomial mutation in numeric space and are rounded to the nearest
    allowed value. Continuous variables use polynomial mutation within bounds.
    """
    mutated_ind = deepcopy(ind)
    if pm is None:
        pm = 1.0 / layout.dim

    for offset, choices in enumerate(categorical_choices):
        if rng.random() > pm:
            continue
        current_value = mutated_ind.solution[offset]
        candidates = [choice for choice in choices if choice != current_value]
        if not candidates:
            continue
        mutated_ind.solution[offset] = rng.choice(candidates).item()

    for idx in range(layout.discrete_slice.start, layout.continuous_slice.stop):
        if rng.random() > pm:
            continue
        if idx < layout.continuous_slice.start:
            values = discrete_values[idx - layout.discrete_slice.start]
            x_low, x_up = min(values), max(values)
        else:
            x_low, x_up = bounds[idx - layout.continuous_slice.start]
        if x_up - x_low <= 0:
            continue
        delta1 = (mutated_ind.solution[idx] - x_low) / (x_up - x_low)
        delta2 = (x_up - mutated_ind.solution[idx]) / (x_up - x_low)
        u = rng.random()
        if u < HALF_PROBABILITY:
            val = 2.0 * u + (1.0 - 2.0 * u) * ((1 - delta1) ** (eta + 1))
            delta = val ** (1.0 / (eta + 1.0)) - 1.0
        else:
            val = 2.0 * (1.0 - u) + 2.0 * (u - HALF_PROBABILITY) * ((1 - delta2) ** (eta + 1))
            delta = 1.0 - val ** (1.0 / (eta + 1.0))
        mutated_ind.solution[idx] = np.clip(mutated_ind.solution[idx] + delta * (x_up - x_low), x_low, x_up)

    return repair_mixed_solution(mutated_ind, layout, categorical_choices, discrete_values, bounds)


def initialize_population(
    population_size: int,
    bounds: list[tuple[float, float]],
    rng: np.random.Generator,
    mean: np.ndarray | None = None,
    init_distribution_eta: float = 2.0,
) -> Population:
    """
    Generate an initial population within bounds or around a center solution.

    Args:
        population_size (int): Number of individuals.
        bounds (list[tuple[float, float]]): Variable bounds.
        rng (np.random.Generator): Random number generator.
        mean (np.ndarray | None): Initial center solution. Defaults to None.
        init_distribution_eta (float): Polynomial mutation eta used when initializing from mean.

    Returns:
        Population: Initial population.
    """
    if mean is None:
        return Population(
            {i: Individual(solution=[rng.uniform(low, high) for (low, high) in bounds]) for i in range(population_size)}
        )
    base_ind = Individual(solution=list(mean))
    return Population(
        {
            i: polynomial_mutation(base_ind, bounds, rng, eta=init_distribution_eta, pm=1.0)
            for i in range(population_size)
        }
    )


def initialize_mixed_population(
    population_size: int,
    layout: VariableLayout,
    categorical_choices: list[list[float | int]],
    discrete_values: list[list[float | int]],
    bounds: list[tuple[float, float]],
    rng: np.random.Generator,
) -> Population:
    """Generate an initial population for a contiguous mixed-variable solution."""

    def sample_solution() -> list[float | int]:
        solution: list[float | int] = []
        solution.extend(rng.choice(choices).item() for choices in categorical_choices)
        solution.extend(rng.choice(values).item() for values in discrete_values)
        solution.extend(rng.uniform(bounds[idx][0], bounds[idx][1]) for idx in range(layout.num_continuous))
        return solution

    return Population({i: Individual(solution=sample_solution()) for i in range(population_size)})
