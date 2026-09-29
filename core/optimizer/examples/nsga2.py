"""
nsga2.py
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

from copy import deepcopy

import numpy as np

from core.optimizer.examples.helpers.nondominant_archive import NonDominatedArchive
from emsopt_engine.individual import Individual, Population
from emsopt_engine.interface.mo_optimizer_base import MOOptimizerBase
from emsopt_engine.interface.optimizer_interface import OptimizerInterface
from emsopt_engine.registry import optimizer


class NSGA2(MOOptimizerBase, OptimizerInterface):
    EPS = 1e-10

    def __init__(
        self,
        dim: int,
        num_obj: int,
        population_size: int | None = None,
        num_children: int | None = None,
        bounds: tuple[float, float] | list[tuple[float, float]] | None = None,
        seed: int | None = None,
    ) -> None:
        self.dim = dim
        self.num_obj = num_obj
        if population_size is None:
            population_size = num_obj * 100
        self.population_size = population_size
        if num_children is None:
            num_children = population_size
        self.n_children = num_children
        if bounds is None:
            bounds = [(-1, 1)] * self.dim
        elif np.asarray(bounds).ndim == 1:
            bounds = [bounds] * self.dim
        elif (np.asarray(bounds).ndim == 2) and (len(bounds) <= dim):
            # fill bounds by (-1, 1)
            for _ in range(dim - len(bounds)):
                bounds.append((-1, 1))
        else:
            raise ValueError("bounds must be a tuple or list of tuples with length equal to or smaller than dim.")
        self.bounds = bounds
        self.seed = seed
        super().__init__(self.population_size, self.seed)
        self.rng = np.random.default_rng(self.seed)
        self._init_population()
        self.archive = NonDominatedArchive(num_obj)

    def _init_population(self) -> None:
        """
        Generate the initial population for the NSGA-II algorithm.
        Each individual is initialized with random values within the specified bounds.
        """
        self.population = Population(
            {
                i: Individual(solution=[self.rng.uniform(low, high) for (low, high) in self.bounds])
                for i in range(self.population_size)
            }
        )

    def setup_population(self, evaluated_population: Population) -> None:
        """
        Set the population from an externally evaluated Population object and update the Pareto front archive.
        Args:
            evaluated_population (Population): The externally evaluated population to set.
        """
        self.population = deepcopy(evaluated_population)
        self.archive.add_population(self.population)
        self.pareto = self.archive.archive

    def get_candidates(self) -> Population:
        """
        Generate offspring population using tournament selection, SBX crossover, and polynomial mutation.
        Returns:
            Population: The generated offspring population.
        """
        offspring = []
        while len(offspring) < self.n_children:
            p1 = self._tournament(self.population)
            p2 = self._tournament(self.population)
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
        fronts = self.fast_nondominated_sort(population.extract(indices))
        if len(fronts[0]) == 1:
            return fronts[0][0]
        crowding = self.crowding_distance(fronts[0])
        return fronts[0][np.argmax(crowding)]

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
        child1, child2 = deepcopy(parent1), deepcopy(parent2)
        for i in range(self.dim):
            if self.rng.random() > pc:
                continue
            x1, x2 = parent1.solution[i], parent2.solution[i]
            x_low, x_up = self.bounds[i]
            if x_up - x_low <= 0:
                continue
            if abs(x1 - x2) > self.EPS:
                u = self.rng.random()
                if u <= 0.5:
                    beta = (2.0 * u) ** (1.0 / (eta + 1.0))
                else:
                    beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                if x1 > x2:
                    x1, x2 = x2, x1
                child1.solution[i] = 0.5 * ((x1 + x2) - beta * abs(x2 - x1))
                child2.solution[i] = 0.5 * ((x1 + x2) + beta * abs(x2 - x1))
                child1.solution[i] = np.clip(child1.solution[i], x_low, x_up)
                child2.solution[i] = np.clip(child2.solution[i], x_low, x_up)
        return child1, child2

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
        mutated_ind = deepcopy(ind)
        if pm is None:
            pm = 1.0 / self.dim
        for i in range(self.dim):
            x_low, x_up = self.bounds[i]
            if x_up - x_low <= 0:
                continue
            if self.rng.random() > pm:
                continue
            delta1 = (ind.solution[i] - x_low) / (x_up - x_low)
            delta2 = (x_up - ind.solution[i]) / (x_up - x_low)
            u = self.rng.random()
            if u < 0.5:
                val = 2.0 * u + (1.0 - 2.0 * u) * (delta1 ** (eta + 1))
                delta = val ** (1.0 / (eta + 1.0)) - 1.0
            else:
                val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (delta2 ** (eta + 1))
                delta = 1.0 - val ** (1.0 / (eta + 1.0))
            mutated_ind.solution[i] = np.clip(ind.solution[i] + delta * (x_up - x_low), x_low, x_up)
        return mutated_ind

    def proceed_to_next_iteration(self, evaluated_candidates: Population) -> None:
        """
        Integrate evaluated offspring with parent population and select the next generation.
        Also updates the Pareto front archive.
        Args:
            evaluated_candidates (Population): The evaluated offspring population.
        """
        self.population.merge(evaluated_candidates)
        fronts = self.fast_nondominated_sort(self.population)
        next_pop = []
        for front in fronts:
            if len(next_pop) + len(front) > self.population_size:
                crowding = self.crowding_distance(front)
                sorted_front = [
                    ind
                    for _, ind in sorted(zip(crowding, front, strict=True), key=lambda zipped: zipped[0], reverse=True)
                ]
                next_pop.extend(sorted_front[: self.population_size - len(next_pop)])
                break
            next_pop.extend(front)
        self.population = Population(dict(enumerate(next_pop)))
        self.archive.add_population(self.population)
        self.pareto = self.archive.archive

    def fast_nondominated_sort(self, population: Population) -> list[list[Individual]]:
        """
        Perform fast non-dominated sorting on the population.
        Args:
            population (Population): The population to sort.
        Returns:
            list[list[Individual]]: List of fronts, each containing individuals.
        """
        f = {idx: np.array(ind.metrics.objectives) for idx, ind in population.items()}
        cv = {idx: ind.metrics.constraint_violation for idx, ind in population.items()}

        s = {p: [q for q in population if self._dominates((f[p], cv[p]), (f[q], cv[q]))] for p in population}
        n = {p: len([q for q in population if self._dominates((f[q], cv[q]), (f[p], cv[p]))]) for p in population}
        fronts = [[]]
        fronts_indices = [[]]
        # find pareto (rank = 0)
        for p in population:
            if n[p] == 0:
                fronts[0].append(population[p])
                fronts_indices[0].append(p)
        # find others
        i = 0
        while fronts_indices[i]:
            next_front = []
            next_front_indices = []
            for p in fronts_indices[i]:
                for q in s[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        next_front.append(population[q])
                        next_front_indices.append(q)
            i += 1
            fronts.append(next_front)
            fronts_indices.append(next_front_indices)
        return [front for front in fronts if front]

    def crowding_distance(self, front: list[Individual]) -> np.ndarray:
        """
        Calculate the crowding distance for each individual in a front.
        Args:
            front (list[Individual]): The front of individuals.
        Returns:
            np.ndarray: Array of crowding distances for the front.
        """
        if not front:
            return []
        num_obj = self.num_obj
        distances = np.zeros(len(front))
        for m in range(num_obj):
            obj_values = np.array([ind.metrics.objectives[m] for ind in front])
            sorted_idx = np.argsort(obj_values)
            distances[sorted_idx[0]] = distances[sorted_idx[-1]] = np.inf
            min_obj = obj_values[sorted_idx[0]]
            max_obj = obj_values[sorted_idx[-1]]
            if max_obj - min_obj == 0:
                continue
            for i in range(1, len(front) - 1):
                distances[sorted_idx[i]] += (obj_values[sorted_idx[i + 1]] - obj_values[sorted_idx[i - 1]]) / (
                    max_obj - min_obj
                )
        return distances

    def _dominates(self, f_cv_p: tuple[np.ndarray, float], f_cv_q: tuple[np.ndarray, float]) -> bool:
        """
        Determine if individual ind_p dominates individual ind_q.
        Args:
            f_cv_p (tuple[np.ndarray, float]): Objectives and constraint violation of individual p.
            f_cv_q (tuple[np.ndarray, float]): Objectives and constraint violation of individual q.
        Returns:
            bool: True if ind_p dominates ind_q, False otherwise.
        """
        f_p, cv_p = f_cv_p
        f_q, cv_q = f_cv_q
        if cv_p > 0 and cv_q > 0:
            return cv_p < cv_q
        if cv_p == 0 and cv_q > 0:
            return True
        if cv_p > 0 and cv_q == 0:
            return False
        return bool(all(f_p <= f_q) and any(f_p < f_q))


@optimizer("nsga2")
def build_nsga2(
    dim: int,
    num_obj: int,
    population_size: int | None = None,
    num_children: int | None = None,
    bounds: tuple[float, float] | list[tuple[float, float]] | None = None,
    seed: int | None = None,
) -> OptimizerInterface:
    """
    NSGA-II multi-objective evolutionary optimization algorithm class including constraint handling.
    Inherits from MOOptimizerBase and OptimizerInterface.

    Args:
        dim (int): optimization problem dimension (Individual size). When Using pyemsol_shape_evaluator, this will be automatically set by EMSOptimizer.
        n_objectives (int): Number of objective functions. When Using pyemsol_shape_evaluator, this will be automatically set by EMSOptimizer.
        population_size (int | None, optional): Population size. If None, defaults to dim * 10.
        n_children (int | None, optional): Number of offspring. If None, defaults to population_size.
        bounds (tuple[float, float] | list[tuple[float, float]] | None , optional): Range for each variable
        seed (int | None, optional): Random seed for reproducibility.
    """
    return NSGA2(
        dim=dim,
        num_obj=num_obj,
        population_size=population_size,
        num_children=num_children,
        bounds=bounds,
        seed=seed,
    )
