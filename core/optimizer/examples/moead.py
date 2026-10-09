"""
moead.py
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

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from core.optimizer.examples.cmaes import CMAES
from core.optimizer.examples.helpers.decomposition import (
    Scalarizer,
    generate_scalarizer,
    generate_weights_dirichlet,
    generate_weights_grid,
    initialize_reference_and_nadir,
    neighbor_indices,
    update_reference_and_nadir,
)
from core.optimizer.examples.helpers.nondominant_archive import NonDominatedArchive
from core.optimizer.examples.helpers.variation import (
    initialize_population,
    normalize_bounds,
    polynomial_mutation,
    sbx_crossover,
)
from emsopt_engine.individual import Individual, Population
from emsopt_engine.interface.mo_optimizer_base import MOOptimizerBase
from emsopt_engine.interface.optimizer_interface import OptimizerInterface
from emsopt_engine.interface.so_optimizer_base import SOOptimizerBase
from emsopt_engine.registry import optimizer


@dataclass
class SubProblem:
    optimizer: SOOptimizerBase
    candidate_indices: list[int]
    weights: list[float]


class DecompositionEnsemble(MOOptimizerBase, OptimizerInterface):
    MAX_VALUE = 1e9
    MIN_NADIR_DIFF = 1e-12

    def __init__(
        self,
        num_obj: int,
        subproblem_weights: list[list[float]],
        scalarizer: Scalarizer,
        optimizer_cls: type[SOOptimizerBase],
        optimizer_args: list | None = None,
        optimizer_kwargs: dict | None = None,
        seed: int | None = None,
    ) -> None:
        if optimizer_args is None:
            optimizer_args = []
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        self.num_obj = num_obj
        # validation
        for i, w in enumerate(subproblem_weights):
            if len(w) != self.num_obj:
                msg = f"weights[{i}] has length {len(w)} != {self.num_obj}"
                raise ValueError(msg)
        # disable seed for soo
        if "seed" in optimizer_kwargs:
            optimizer_kwargs["seed"] = None
        # initialize members
        _optimizer = optimizer_cls(*optimizer_args, **optimizer_kwargs)  # just for population size
        self.num_subproblems = len(subproblem_weights)
        self.subproblem_population_size = _optimizer.population_size
        population_size = self.subproblem_population_size * self.num_subproblems
        super().__init__(population_size, seed)
        self.scalarizer = scalarizer
        self._z_ref: list[float] | None = None
        self._z_nadir: list[float] | None = None
        self.archive = NonDominatedArchive(self.num_obj)
        # create random number generator
        rng = np.random.default_rng(seed)
        # define sub problems
        self.sub_problems = [
            SubProblem(
                optimizer_cls(*optimizer_args, **optimizer_kwargs, seed=int(rng.integers(0, 10000))),
                list(range(pop_slice, pop_slice + self.subproblem_population_size)),
                list(subproblem_weights[idx]),
            )
            for idx, pop_slice in enumerate(range(0, self.population_size, self.subproblem_population_size))
        ]
        # constract init population
        self.population = self._construct_population()

    def _construct_population(self) -> Population:
        """Construct initial population for all subproblems"""
        population = Population()
        for sub_problem in self.sub_problems:
            if sub_problem.optimizer.population is not None:
                population.merge(sub_problem.optimizer.population)
        return population

    def _update_pareto(self, population: Population) -> None:
        """Update pareto front from archive"""
        self.archive.add_population(population)
        self.pareto = self.archive.archive

    def _initialize_reference_and_nadir(self) -> None:
        self._z_ref, self._z_nadir = initialize_reference_and_nadir(self.num_obj, self._z_ref, self._z_nadir)

    def _update_reference_and_nadir(self) -> None:
        """Update reference and nadir points from archive"""
        self._z_ref, self._z_nadir = update_reference_and_nadir(
            archive=self.archive.archive,
            num_objectives=self.num_obj,
            scalarizer=self.scalarizer,
            reference=self._z_ref,
            nadir=self._z_nadir,
            max_value=self.MAX_VALUE,
            min_nadir_diff=self.MIN_NADIR_DIFF,
        )

    def setup_population(self, evaluated_population: Population) -> None:
        """Call setup_population of optimizers

        Args:
            population (Population): population for setup
        """
        for sub_problem in self.sub_problems:
            sub_pop = evaluated_population.extract(sub_problem.candidate_indices)
            sub_pop.reindex()
            sub_problem.optimizer.setup_population(sub_pop)
        self._update_pareto(evaluated_population)

    def get_candidates(self) -> Population:
        """
        Get candidates from current optimizers
        Returns:
            Population: Population of individuals
        Raises:
            Exception: If an error occurs during population generation
        """
        candidates = Population()
        for sub_problem in self.sub_problems:
            subproblem_candidates = sub_problem.optimizer.get_candidates()
            candidates.merge(subproblem_candidates)
        return candidates

    def proceed_to_next_iteration(self, evaluated_candidates: Population) -> None:
        """
        Update optimizers state based on evaluated_candidates
        AND update pareto (as external population in context of MOEA/D)
        Args:
            evaluated_candidates: Candidate population with evaluation values
        """
        # update pareto front
        self._update_pareto(evaluated_candidates)
        self._update_reference_and_nadir()
        for sub_problem in self.sub_problems:
            sub_cans = evaluated_candidates.extract(sub_problem.candidate_indices)
            sub_cans.reindex()
            # convert muiti-objective problem into decomposed single-objective subproblem
            for ind in sub_cans.values():
                ind.metrics.fitness = self.scalarizer(ind.metrics.objectives, sub_problem.weights)
            sub_problem.optimizer.proceed_to_next_iteration(sub_cans)
        self.population = evaluated_candidates


class MOEAD(MOOptimizerBase, OptimizerInterface):
    """
    MOEA/D optimizer using neighboring decomposition subproblems.

    This implementation follows the standard MOEA/D loop: one incumbent solution
    is maintained for each weight vector, parents are sampled from neighboring
    subproblems, and evaluated offspring replace neighboring incumbents when they
    improve the corresponding scalarized objective.
    """

    EPS = 1e-10
    MAX_VALUE = 1e9
    MIN_NADIR_DIFF = 1e-12
    CHILD_SELECTION_PROBABILITY = 0.5

    def __init__(
        self,
        dim: int,
        num_obj: int,
        subproblem_weights: list[list[float]] | np.ndarray,
        neighborhood_size: int = 20,
        decomposition_type: str = "tchebycheff",
        bounds: tuple[float, float] | list[tuple[float, float]] | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Initialize MOEA/D population, weight vectors, neighborhoods, and archive.

        Args:
            dim (int): Optimization problem dimension (Individual size).
                When using pyemsol_shape_evaluator, this is automatically set by EMSOptimizer.
            num_obj (int): Number of objectives.
                When using pyemsol_shape_evaluator, this is automatically set by EMSOptimizer.
            subproblem_weights (list[list[float]] | np.ndarray): Decomposition weights
                with shape (N, num_obj).
            neighborhood_size (int): Number of neighboring weight vectors for mating and update.
                The value is clipped to the number of subproblems. Defaults to 20.
            decomposition_type (str): Scalarizing function name
                ("weighted_sum", "tchebycheff", or "pbi"). Defaults to "tchebycheff".
            bounds (tuple[float, float] | list[tuple[float, float]] | None, optional):
                Variable bounds. If None, [-1, 1] is used for each dimension.
            seed (int | None, optional): Random seed for reproducibility. Defaults to None.

        Raises:
            ValueError: If subproblem_weights or neighborhood_size is invalid.
        """
        self.dim = dim
        self.num_obj = num_obj
        self.bounds = normalize_bounds(bounds, self.dim)
        self.seed = seed
        self.weights = np.asarray(subproblem_weights, dtype=float)
        if self.weights.ndim != 2 or self.weights.shape[1] != num_obj or self.weights.shape[0] < 1:  # noqa: PLR2004
            msg = "subproblem_weights must be shape (N, num_obj) with N>=1"
            raise ValueError(msg)
        population_size = len(self.weights)
        resolved_neighborhood_size = min(self._validate_neighborhood_size(neighborhood_size), population_size)
        super().__init__(population_size, self.seed)
        self.rng = np.random.default_rng(self.seed)
        self.num_decomposition = population_size
        self.neighborhood_size = resolved_neighborhood_size
        self.neighborhoods = neighbor_indices(self.weights, self.neighborhood_size)
        self.scalarizer = generate_scalarizer(decomposition_type)
        self.archive = NonDominatedArchive(num_obj)
        self.population = initialize_population(self.population_size, self.bounds, self.rng)
        self._z_ref: list[float] | None = None
        self._z_nadir: list[float] | None = None
        self._pending_candidate_targets: list[int] = []

    @staticmethod
    def _validate_neighborhood_size(neighborhood_size: int) -> int:
        """
        Validate the MOEA/D neighborhood size.

        Args:
            neighborhood_size (int): Requested number of neighboring subproblems.

        Returns:
            int: Validated neighborhood size.

        Raises:
            ValueError: If neighborhood_size is smaller than 1.
        """
        if neighborhood_size < 1:
            msg = "neighborhood_size must be >= 1"
            raise ValueError(msg)
        return neighborhood_size

    def _update_reference_and_nadir(self, evaluated_candidates: Population | None = None) -> None:
        """
        Update reference/nadir points from incumbents, candidates, and archive.

        Args:
            evaluated_candidates (Population | None): Newly evaluated candidates to include
                in the reference/nadir update. Defaults to None.
        """
        reference_source = Population()
        if self.population is not None:
            reference_source.merge(self.population)
        if evaluated_candidates is not None:
            reference_source.merge(evaluated_candidates)
        reference_source.merge(self.archive.archive)
        self._z_ref, self._z_nadir = update_reference_and_nadir(
            archive=reference_source,
            num_objectives=self.num_obj,
            scalarizer=self.scalarizer,
            reference=self._z_ref,
            nadir=self._z_nadir,
            max_value=self.MAX_VALUE,
            min_nadir_diff=self.MIN_NADIR_DIFF,
        )

    def _is_better_for_subproblem(self, candidate: Individual, incumbent: Individual, weights: np.ndarray) -> bool:
        """
        Compare candidate and incumbent for one decomposed subproblem.

        Feasible solutions are preferred over infeasible ones. If both are feasible,
        the selected scalarizer is used. If both are infeasible, the smaller
        constraint violation is preferred.

        Args:
            candidate (Individual): Newly evaluated candidate.
            incumbent (Individual): Current solution assigned to the subproblem.
            weights (np.ndarray): Weight vector for the subproblem.

        Returns:
            bool: True if candidate should replace incumbent.
        """
        candidate_cv = candidate.metrics.constraint_violation
        incumbent_cv = incumbent.metrics.constraint_violation
        candidate_feasible = candidate_cv < self.EPS
        incumbent_feasible = incumbent_cv < self.EPS
        if candidate_feasible and not incumbent_feasible:
            return True
        if not candidate_feasible and incumbent_feasible:
            return False
        if not candidate_feasible and not incumbent_feasible:
            return candidate_cv < incumbent_cv
        candidate_value = self.scalarizer(candidate.metrics.objectives, weights)
        incumbent_value = self.scalarizer(incumbent.metrics.objectives, weights)
        return candidate_value < incumbent_value

    def setup_population(self, evaluated_population: Population) -> None:
        """
        Set the initial evaluated population and update the external archive.

        Args:
            evaluated_population (Population): Initial population with objective values.

        Raises:
            ValueError: If evaluated_population size does not match the number of subproblems.
        """
        if len(evaluated_population) != self.population_size:
            msg = f"evaluated_population size {len(evaluated_population)} != population_size {self.population_size}"
            raise ValueError(msg)
        self.population = deepcopy(evaluated_population)
        self.population.reindex()
        self.archive.add_population(self.population)
        self._update_reference_and_nadir()
        self.pareto = self.archive.archive

    def get_candidates(self) -> Population:
        """
        Generate one offspring candidate for each decomposition subproblem.

        Parents are selected from the current subproblem neighborhood, then SBX crossover
        and polynomial mutation are applied using shared variation operators.

        Returns:
            Population: Candidate population to be evaluated.
        """
        candidates = Population()
        self._pending_candidate_targets = []
        for subproblem_idx in range(self.population_size):
            neighbor_ids = self.neighborhoods[subproblem_idx]
            replace = len(neighbor_ids) < 2  # noqa: PLR2004
            parent_ids = self.rng.choice(neighbor_ids, size=2, replace=replace)
            parent1 = self.population[int(parent_ids[0])]
            parent2 = self.population[int(parent_ids[1])]
            child1, child2 = sbx_crossover(parent1, parent2, self.bounds, self.rng, eps=self.EPS)
            child = child1 if self.rng.random() < self.CHILD_SELECTION_PROBABILITY else child2
            candidates[subproblem_idx] = polynomial_mutation(child, self.bounds, self.rng)
            self._pending_candidate_targets.append(subproblem_idx)
        return candidates

    def proceed_to_next_iteration(self, evaluated_candidates: Population) -> None:
        """
        Update archive, reference/nadir points, and neighboring subproblem incumbents.

        Args:
            evaluated_candidates (Population): Evaluated offspring population.
        """
        self.archive.add_population(evaluated_candidates)
        self._update_reference_and_nadir(evaluated_candidates)
        candidate_targets = self._pending_candidate_targets or list(evaluated_candidates.keys())
        for candidate, target_idx in zip(evaluated_candidates.values(), candidate_targets, strict=False):
            # replace with candidate if it is the best among target_idx subproblem's neighborhoods
            for raw_neighbor_idx in self.neighborhoods[target_idx]:
                neighbor_idx = int(raw_neighbor_idx)
                if self._is_better_for_subproblem(candidate, self.population[neighbor_idx], self.weights[neighbor_idx]):
                    self.population[neighbor_idx] = deepcopy(candidate)
        self.population.reindex()
        self.pareto = self.archive.archive
        self._pending_candidate_targets = []


@optimizer("decomposition_ensemble")
def build_decomposition_ensemble(
    dim: int,
    num_obj: int,
    num_decomposition: int = 10,
    decomposition_type: str = "tchebycheff",
    seed: int | None = None,
    mean: np.ndarray | None = None,
    sigma: float = 1.0,
    bounds: tuple[float, float] | list[tuple[float, float]] | None = None,
    population_size: int | None = None,
) -> OptimizerInterface:
    """Simple decomposition ensemble algorithm
    - Subproblems are gereneted by a specified decomposition strategy (called "scalarizer")
    - Each subproblem is assigned to single CMA-ES optimizer instance
    - Neighbor subproblem information does not used for any processes.
      Instead, this class just solve independent subproblems by independent instances of CMA-ES
    - Pareto (Non-dominant) solutions are stored, considering constraint values

    Args:
        dim (int): optimization problem dimension (Individual size).
            When Using pyemsol_shape_evaluator, this will be automatically set by EMSOptimizer.
        num_obj (int): number of objectives.
            When Using pyemsol_shape_evaluator, this will be automatically set by EMSOptimizer.
        num_decomposition (int): number of decomposition (subproblems)
        decomposition_type (str): decomposition type ("weighted_sum" or "tchebycheff" or "pbi").
            Defaults to "tchebycheff".
        seed (int | None, optional): random seed. Defaults to None.
        mean (np.ndarray | None, optional): Initial mean vector for CMA-ES.
            Raises error if dimension does not match individual_size. Defaults to None.
        sigma (float, optional): Initial standard deviation for CMA-ES. Defaults to 1.0.
        bounds (tuple[float, float] | list[tuple[float, float]] | None, optional): Bounds for CMA-ES. Defaults to None.
        population_size (int | None, optional): Population size for CMA-ES. Defaults to None.
    """
    weights = generate_weights_dirichlet(num_obj, num_decomposition, seed)
    scalarizer = generate_scalarizer(decomposition_type)
    args = [dim]
    kwargs = {"mean": mean, "sigma": sigma, "bounds": bounds, "population_size": population_size}
    return DecompositionEnsemble(num_obj, weights, scalarizer, CMAES, args, kwargs, seed=seed)


@optimizer("moead")
def build_moead(
    dim: int,
    num_obj: int,
    num_grid_division: int = 10,
    neighborhood_size: int = 10,
    decomposition_type: str = "tchebycheff",
    bounds: tuple[float, float] | list[tuple[float, float]] | None = None,
    seed: int | None = None,
) -> OptimizerInterface:
    """
    MOEA/D optimizer based on neighboring decomposition subproblems.

    Args:
        dim (int): Optimization problem dimension (Individual size).
            When using pyemsol_shape_evaluator, this is automatically set by EMSOptimizer.
        num_obj (int): Number of objectives.
            When using pyemsol_shape_evaluator, this is automatically set by EMSOptimizer.
        num_grid_division (int): Number of grid division. Defaults to 10.
        neighborhood_size (int): Number of neighboring subproblems. Defaults to 10.
        decomposition_type (str): Scalarizer name ("weighted_sum", "tchebycheff", or "pbi").
            Defaults to "tchebycheff".
        bounds (tuple[float, float] | list[tuple[float, float]] | None, optional):
            Variable bounds. Defaults to None.
        seed (int | None, optional): Random seed. Defaults to None.
    """
    weights = generate_weights_grid(num_obj, num_grid_division)
    return MOEAD(
        dim=dim,
        num_obj=num_obj,
        subproblem_weights=weights,
        neighborhood_size=neighborhood_size,
        decomposition_type=decomposition_type,
        bounds=bounds,
        seed=seed,
    )
