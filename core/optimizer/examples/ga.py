"""
ga.py
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

from collections.abc import Callable
from copy import deepcopy
from logging import getLogger

import numpy as np

from core.optimizer.examples.helpers.decomposition import (
    Scalarizer,
    generate_scalarizer,
    initialize_reference_and_nadir,
    update_reference_and_nadir,
)
from core.optimizer.examples.helpers.variation import (
    VariableTypeCounts,
    initialize_mixed_population,
    initialize_population,
    mixed_crossover,
    mixed_mutation,
    normalize_bounds,
    normalize_choice_values,
    normalize_variable_layout,
    polynomial_mutation,
    sbx_crossover,
)
from emsopt_engine.individual import Individual, Population
from emsopt_engine.interface.optimizer_interface import OptimizerInterface
from emsopt_engine.interface.so_optimizer_base import SOOptimizerBase
from emsopt_engine.registry import optimizer

logger = getLogger(__name__)


class GA(SOOptimizerBase, OptimizerInterface):
    EPS = 1e-10

    @staticmethod
    def _resolve_population_size(population_size: int | None, dim: int) -> int:
        if population_size is None:
            return dim * 10
        return population_size

    @staticmethod
    def _resolve_num_parents(num_parents: int | None, population_size: int) -> int:
        if num_parents is None:
            return population_size
        return num_parents

    @staticmethod
    def _resolve_num_children(num_children: int | None, num_parents: int) -> int:
        if num_children is None:
            return 2 * num_parents
        return num_children

    @staticmethod
    def _normalize_bounds(
        bounds: tuple[float, float] | list[tuple[float, float]] | None,
        dim: int,
    ) -> list[tuple[float, float]]:
        return normalize_bounds(bounds, dim)

    def __init__(  # noqa: PLR0913
        self,
        dim: int,
        num_obj: int = 1,
        population_size: int | None = None,
        num_parents: int | None = None,
        num_children: int | None = None,
        bounds: tuple[float, float] | list[tuple[float, float]] | None = None,
        variable_type_counts: VariableTypeCounts | None = None,
        categorical_choices: list[list[float | int]] | None = None,
        discrete_values: list[list[float | int]] | None = None,
        seed: int | None = None,
        scalarizer: Scalarizer | None = None,
        scalarizer_weights: list[float] | np.ndarray | None = None,
    ) -> None:
        if not isinstance(dim, int) or dim <= 0:
            msg = "dim must be a positive integer"
            raise ValueError(msg)
        if not isinstance(num_obj, int) or num_obj <= 0:
            msg = "num_obj must be a positive integer"
            raise ValueError(msg)
        self.dim = dim
        self.num_obj = num_obj
        self.population_size = self._resolve_population_size(population_size, self.dim)
        self.n_parents = self._resolve_num_parents(num_parents, self.population_size)
        self.n_children = self._resolve_num_children(num_children, self.n_parents)
        if self.n_parents > self.population_size:
            msg = "num parents must be equal to or smaller than population size."
            raise ValueError(msg)
        self.variable_layout = normalize_variable_layout(variable_type_counts, self.dim)
        self.categorical_choices = normalize_choice_values(
            categorical_choices,
            self.variable_layout.num_categorical,
            "categorical_choices",
        )
        self.discrete_values = normalize_choice_values(
            discrete_values,
            self.variable_layout.num_discrete,
            "discrete_values",
        )
        self.bounds = self._normalize_bounds(bounds, self.variable_layout.num_continuous)
        self.seed = seed
        super().__init__(self.population_size, self.seed)
        self.rng = np.random.default_rng(self.seed)
        self.scalarizer = scalarizer
        self.scalarizer_weights = scalarizer_weights
        self._z_ref: list[float] | None = None
        self._z_nadir: list[float] | None = None
        if self.variable_layout.is_continuous_only:
            self.population = initialize_population(self.population_size, self.bounds, self.rng)
        else:
            self.population = initialize_mixed_population(
                self.population_size,
                self.variable_layout,
                self.categorical_choices,
                self.discrete_values,
                self.bounds,
                self.rng,
            )

    def _initialize_reference_and_nadir(self) -> None:
        self._z_ref, self._z_nadir = initialize_reference_and_nadir(self.num_obj, self._z_ref, self._z_nadir)

    def _update_reference_and_nadir(self, population: Population) -> None:
        self._z_ref, self._z_nadir = update_reference_and_nadir(
            archive=population,
            num_objectives=self.num_obj,
            scalarizer=self.scalarizer,
            reference=self._z_ref,
            nadir=self._z_nadir,
        )

    def _fitness_with_contrait(self, worst_fitness: float) -> Callable:
        def inner(ind: Individual) -> float:
            if ind.metrics.constraint_violation > self.EPS:
                return worst_fitness + ind.metrics.constraint_violation
            return ind.metrics.fitness

        return inner

    def setup_population(self, evaluated_population: Population) -> None:
        """
        Set the population from an externally evaluated Population object.
        Args:
            evaluated_population (Population): The externally evaluated population to set.
        """
        self.population = deepcopy(evaluated_population)

    def get_candidates(self) -> Population:
        """
        Generate offspring population using tournament selection, SBX crossover, and polynomial mutation.
        Returns:
            Population: The generated offspring population.
        """
        indices = self.rng.choice(list(self.population.keys()), size=self.n_parents, replace=False)
        parents, self.population = self.population.split(indices)  # Just Generation Gap
        offspring = []
        while len(offspring) < self.n_children:
            p1 = self._tournament(parents)
            p2 = self._tournament(parents)
            c1, c2 = self._sbx_crossover(p1, p2)
            c1 = self._polynomial_mutation(c1)
            c2 = self._polynomial_mutation(c2)
            offspring.extend([c1, c2])
        return Population(dict(enumerate(offspring[: self.n_children])))

    def _tournament(self, population: Population, k: int = 2) -> Individual:
        """
        Perform tournament selection on the population.
        Args:
            population (Population): The population to select from.
            k (int): Number of individuals to participate in the tournament.
        Returns:
            Individual: The selected individual.
        """
        indices = self.rng.choice(list(population.keys()), size=k, replace=False)
        # tournament selection
        participants_list = list(population.extract(indices).values())
        worst_fitness = max(ind.metrics.fitness for ind in participants_list)
        return sorted(participants_list, key=self._fitness_with_contrait(worst_fitness))[0]

    def _sbx_crossover(
        self, parent1: Individual, parent2: Individual, eta: float = 20, pc: float = 0.75
    ) -> tuple[Individual, Individual]:
        """
        Perform Simulated Binary Crossover (SBX) between two parents.
        Args:
            parent1 (Individual): The first parent individual.
            parent2 (Individual): The second parent individual.
            eta (float): Distribution index for SBX.
            pm (float): Crossover probability. Defaults to 1/dim.
        Returns:
            tuple[Individual, Individual]: Two offspring individuals.
        """
        if not self.variable_layout.is_continuous_only:
            return mixed_crossover(
                parent1,
                parent2,
                self.variable_layout,
                self.categorical_choices,
                self.discrete_values,
                self.bounds,
                self.rng,
                eta=eta,
                pc=pc,
                eps=self.EPS,
            )
        return sbx_crossover(parent1, parent2, self.bounds, self.rng, eta=eta, pc=pc, eps=self.EPS)

    def _polynomial_mutation(self, ind: Individual, eta: float = 20, pm: float | None = None) -> Individual:
        """
        Apply polynomial mutation to an individual.
        Args:
            ind (Individual): The individual to mutate.
            eta (float): Distribution index for mutation.
            pm (float | None): Mutation probability. Defaults to 1/dim.
        Returns:
            Individual: The mutated individual.
        """
        if not self.variable_layout.is_continuous_only:
            return mixed_mutation(
                ind,
                self.variable_layout,
                self.categorical_choices,
                self.discrete_values,
                self.bounds,
                self.rng,
                eta=eta,
                pm=pm,
            )
        return polynomial_mutation(ind, self.bounds, self.rng, eta=eta, pm=pm)

    def proceed_to_next_iteration(self, evaluated_candidates: Population) -> None:
        """
        Integrate evaluated offspring with parent population and select the next generation.
        Also updates the Pareto front archive.
        Args:
            evaluated_candidates (Population): The evaluated offspring population.
        """
        self.population.merge(evaluated_candidates)
        if self.scalarizer is not None:
            if self.scalarizer_weights is None:
                msg = "scalarizer_weights must be provided when scalarizer is enabled"
                raise ValueError(msg)
            self._update_reference_and_nadir(self.population)
            for ind in self.population.values():
                ind.metrics.fitness = self.scalarizer(ind.metrics.objectives, self.scalarizer_weights)
        # using elite strategy: best n candidates are next pop
        candidates_list = list(self.population.values())
        worst_fitness = max(ind.metrics.fitness for ind in candidates_list)
        candidates_list.sort(key=self._fitness_with_contrait(worst_fitness))
        next_pop = candidates_list[: self.population_size]
        self.population = Population(dict(enumerate(next_pop)))
        self.update_best_individual()


@optimizer("ga")
def build_ga(  # noqa: PLR0913
    dim: int,
    num_obj: int,
    population_size: int | None = None,
    num_parents: int | None = None,
    num_children: int | None = None,
    bounds: tuple[float, float] | list[tuple[float, float]] | None = None,
    variable_type_counts: VariableTypeCounts | None = None,
    categorical_choices: list[list[float | int]] | None = None,
    discrete_values: list[list[float | int]] | None = None,
    seed: int | None = None,
    scalarizer_type: str | None = None,
    scalarizer_weights: list[float] | np.ndarray | None = None,
) -> OptimizerInterface:
    """
    NSGA-II multi-objective evolutionary optimization algorithm class including constraint handling.
    Inherits from MOOptimizerBase and OptimizerInterface.

    Args:
        dim (int): optimization problem dimension (Individual size).
            When Using pyemsol_shape_evaluator, this will be automatically set by EMSOptimizer.
        num_obj (int): Number of objectives.
            When Using pyemsol_shape_evaluator, this will be automatically set by EMSOptimizer.
        population_size (int | None, optional): Population size. If None, defaults to dim * 10.
        num_parents (int | None, optional): Number of parents. If None, defaults to population_size.
        num_children (int | None, optional): Number of offspring. If None, defaults to 2 * num_parents.
        bounds (tuple[float, float] | list[tuple[float, float]] | None , optional): Range for each variable
        variable_type_counts (dict | None): Counts for categorical, discrete, and continuous variables.
            If None, all variables are treated as continuous. Variables are ordered as
            categorical, discrete, then continuous in Individual.solution.
        categorical_choices (list[list[float | int]] | None): Allowed values for categorical variables.
        discrete_values (list[list[float | int]] | None): Allowed values for discrete variables.
        seed (int | None, optional): Random seed for reproducibility.
        scalarizer_type (str | None): scalarizer type that scalarize multiple objectives (metrics.objectives).
            If None, metrics.fitness is directly used as single objective. Defaults to None.
        scalarizer_weights (list[float] | np.ndarray | None): Weight coefficients used by the scalarizer.
            Required when scalarizer_type is specified.
    """
    scalarizer = generate_scalarizer(scalarizer_type) if scalarizer_type is not None else None
    return GA(
        dim=dim,
        num_obj=num_obj,
        population_size=population_size,
        num_parents=num_parents,
        num_children=num_children,
        bounds=bounds,
        variable_type_counts=variable_type_counts,
        categorical_choices=categorical_choices,
        discrete_values=discrete_values,
        seed=seed,
        scalarizer=scalarizer,
        scalarizer_weights=scalarizer_weights,
    )
