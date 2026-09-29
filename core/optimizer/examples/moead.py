"""
moead.py
The MIT License (MIT)
Copyright © 2025 Sicence Solutions International Laboratory, Inc.

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

from dataclasses import dataclass

import numpy as np

from core.optimizer.examples.cmaes import CMAES
from core.optimizer.examples.helpers.decomposition import (
    PBI,
    Scalarizer,
    Tchebycheff,
    WeightedSum,
    generate_weights_dirichlet,
    generate_weights_grid,
)
from core.optimizer.examples.helpers.nondominant_archive import NonDominatedArchive
from emsopt_engine.individual import Population
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
        # define sub problems
        self.sub_problems = [
            SubProblem(
                optimizer_cls(*optimizer_args, **optimizer_kwargs, seed=np.random.randint(0, 10000)),
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

    def _update_reference_and_nadir(self) -> None:
        """Update reference and nadir points from archive"""
        if self._z_ref is None:
            self._z_ref = [float("inf")] * self.num_obj
        if self._z_nadir is None:
            self._z_nadir = [-float("inf")] * self.num_obj

        # obtain max/min of each objective
        cur_min = [float("inf")] * self.num_obj
        cur_max = [-float("inf")] * self.num_obj
        for ind in self.archive.archive.values():
            fs = ind.metrics.objectives
            for i, f in enumerate(fs):
                cur_min[i] = min(cur_min[i], f)
                if f < self.MAX_VALUE:  # ignore outlier
                    cur_max[i] = max(cur_max[i], f)

        # update z_ref
        eps = 1e-12
        for i in range(self.num_obj):
            if cur_min[i] < self._z_ref[i] - eps:
                self._z_ref[i] = cur_min[i]

        # update z_nadir
        for i in range(self.num_obj):
            if cur_max[i] > self._z_nadir[i] - eps:
                self._z_nadir[i] = cur_max[i]

        for i in range(self.num_obj):
            if self._z_nadir[i] - self._z_ref[i] < 1e-12:
                self._z_nadir[i] = self._z_ref[i] + 1e-12

        self.scalarizer.reference = self._z_ref
        self.scalarizer.nadir = self._z_nadir

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
        dim (int): optimization problem dimension (Individual size). When Using pyemsol_shape_evaluator, this will be automatically set by EMSOptimizer.
        num_obj (int): number of objectives. When Using pyemsol_shape_evaluator, this will be automatically set by EMSOptimizer.
        num_decomposition (int): number of decomposition (subproblems)
        decomposition_type (str): decomposition type ("weighted_sum" or "tchebycheff" or "pbi"). Default is "tchebycheff".
        seed (int | None, optional): random seed. Defaults to None.
        mean (np.ndarray | None, optional): Initial mean vector for CMA-ES. Raises error if dimension does not match individual_size. Defaults to None.
        sigma (float, optional): Initial standard deviation for CMA-ES. Defaults to 1.0.
        bounds (tuple[float, float] | list[tuple[float, float]] | None, optional): Bounds for CMA-ES. Defaults to None.
        population_size (int | None, optional): Population size for CMA-ES. Defaults to None.
    """
    weights = generate_weights_dirichlet(num_obj, num_decomposition, seed)
    if decomposition_type.lower() == "pbi":
        scalarizer = PBI()
    elif decomposition_type.lower() == "weighted_sum":
        scalarizer = WeightedSum()
    elif decomposition_type.lower() == "tchebycheff":
        scalarizer = Tchebycheff()
    else:
        msg = f"Unknown decomposition_type: {decomposition_type}"
        raise ValueError(msg)
    args = [dim]
    kwargs = {"mean": mean, "sigma": sigma, "bounds": bounds, "population_size": population_size}
    return DecompositionEnsemble(num_obj, weights, scalarizer, CMAES, args, kwargs, seed=seed)
