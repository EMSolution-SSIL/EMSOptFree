"""
nsga2.py
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
from logging import getLogger

import numpy as np

from core.optimizer.examples.helpers.nondominant_archive import NonDominatedArchive
from core.optimizer.examples.helpers.variation import (
    initialize_population,
    normalize_bounds,
    normalize_mean,
    polynomial_mutation,
    sbx_crossover,
)
from emsopt_engine.individual import Individual, Population
from emsopt_engine.interface.mo_optimizer_base import MOOptimizerBase
from emsopt_engine.interface.optimizer_interface import OptimizerInterface
from emsopt_engine.registry import optimizer

logger = getLogger(__name__)


class NSGA2(MOOptimizerBase, OptimizerInterface):
    EPS = 1e-10

    @staticmethod
    def _resolve_population_size(population_size: int | None, num_obj: int) -> int:
        if population_size is None:
            return num_obj * 100
        return population_size

    @staticmethod
    def _resolve_num_children(num_children: int | None, population_size: int) -> int:
        if num_children is None:
            return population_size
        return num_children

    @staticmethod
    def _normalize_bounds(
        bounds: tuple[float, float] | list[tuple[float, float]] | None,
        dim: int,
    ) -> list[tuple[float, float]]:
        return normalize_bounds(bounds, dim)

    def _configure_reference_preferences(
        self,
        reference_points: np.ndarray | None,
        reference_weights: np.ndarray | None,
        epsilon: float,
    ) -> None:
        if reference_points is None:
            self.reference_points = None
            self.reference_weights = None
            self.epsilon = float(epsilon)
            return
        ref = np.asarray(reference_points, dtype=float)
        if ref.ndim != 2 or ref.shape[1] != self.num_obj or ref.shape[0] < 1:  # noqa: PLR2004
            msg = "reference_points must be shape (K, num_obj) with K>=1"
            raise ValueError(msg)
        self.reference_points = ref
        if reference_weights is None:
            self.reference_weights = np.ones_like(ref)
        else:
            w = np.asarray(reference_weights, dtype=float)
            if w.shape != ref.shape or np.any(w <= 0):
                msg = "reference_weights must be positive and same shape as reference_points"
                raise ValueError(msg)
            self.reference_weights = w
        self.epsilon = float(epsilon)

    def __init__(  # noqa: PLR0913
        self,
        dim: int,
        num_obj: int,
        population_size: int | None = None,
        num_children: int | None = None,
        bounds: tuple[float, float] | list[tuple[float, float]] | None = None,
        seed: int | None = None,
        mean: np.ndarray | None = None,
        reference_points: np.ndarray | None = None,
        reference_weights: np.ndarray | None = None,
        epsilon: float = 1e-3,
        init_distribution_eta: float = 2.0,
    ) -> None:
        self.dim = dim
        self.num_obj = num_obj
        self.population_size = self._resolve_population_size(population_size, num_obj)
        self.n_children = self._resolve_num_children(num_children, self.population_size)
        self.bounds = self._normalize_bounds(bounds, self.dim)
        self.seed = seed
        self.init_distribution_eta = float(init_distribution_eta)
        if self.init_distribution_eta <= 0:
            msg = "init_distribution_eta must be positive."
            raise ValueError(msg)
        super().__init__(self.population_size, self.seed)
        self.rng = np.random.default_rng(self.seed)
        self._init_population(mean)
        self.archive = NonDominatedArchive(num_obj)
        self._configure_reference_preferences(reference_points, reference_weights, epsilon)

    def _init_population(self, mean: np.ndarray | None = None) -> None:
        """
        Generate the initial population for the NSGA-II algorithm.
        Each individual is initialized with random values within the specified bounds.
        """
        if mean is None:
            self.population = initialize_population(self.population_size, self.bounds, self.rng)
            return

        mean = normalize_mean(mean, self.dim)
        base_ind = Individual(solution=list(mean))
        self.population = Population(
            {
                i: self._polynomial_mutation(base_ind, eta=self.init_distribution_eta, pm=1.0)
                for i in range(self.population_size)
            }
        )

    def _extract_objectives(self, inds: list[Individual]) -> np.ndarray:
        return np.array([ind.metrics.objectives for ind in inds], dtype=float)

    def _minmax_on(self, inds: list[Individual]) -> tuple[np.ndarray, np.ndarray]:
        f = self._extract_objectives(inds)
        fmin = f.min(axis=0)
        fmax = f.max(axis=0)
        # avoid zero width
        span = np.where(fmax > fmin, fmax - fmin, 1.0)
        return fmin, fmin + span

    def _normalize(self, f: np.ndarray, fmin: np.ndarray, fmax: np.ndarray) -> np.ndarray:
        span = np.where(fmax > fmin, fmax - fmin, 1.0)
        return (f - fmin) / span

    def _preference_distance(self, inds: list[Individual], fmin: np.ndarray, fmax: np.ndarray) -> np.ndarray:
        """
        Compute preference distance:
        - For each reference point: weighted Euclidean distance in normalized space
        """
        if not inds:
            return np.array([])
        # N: number of individuals, M: number of objectives, K: number of reference points
        ind_objectives = self._normalize(self._extract_objectives(inds), fmin, fmax)  # (N, M)
        normalized_references = self._normalize(self.reference_points, fmin, fmax)  # (K, M)

        # distances per reference: (K, N)
        # sqrt(sum_i w_i * (f_i - z_i)^2)
        diffs = ind_objectives[None, :, :] - normalized_references[:, None, :]  # (K, N, M)
        dist2_array = (self.reference_weights[:, None, :] * diffs * diffs).sum(axis=2)  # (K, N)
        return np.sqrt(dist2_array)

    def _preference_rank(self, inds: list[Individual], fmin: np.ndarray, fmax: np.ndarray) -> np.ndarray:
        """
        Compute preference distance rank per individual:
        - For each reference point: weighted Euclidean distance in normalized space
        - Rank individuals by distance per reference (1..|set|)
        - Individual score = min_k rank_k
        """
        dist_array = self._preference_distance(inds, fmin, fmax)  # (K, N)

        # ranks per reference (argsort gives indices; we need 1..N ranks)
        ranks = np.zeros_like(dist_array, dtype=int)
        for k in range(self.reference_points.shape[0]):
            order = np.argsort(dist_array[k])  # ascending w.r.t. k-th reference
            # assign ranks 1..N
            r = np.empty_like(order)
            r[order] = np.arange(1, len(inds) + 1)
            ranks[k] = r

        # preference distance = min over refs
        return ranks.min(axis=0)  # (N,)

    def _epsilon_clearing_select(
        self,
        front: list[Individual],
        needed: int,
        fmin: np.ndarray,
        fmax: np.ndarray,
    ) -> list[Individual]:
        if not front or needed <= 0:
            return []
        front_size = len(front)
        pref_rank = self._preference_rank(front, fmin, fmax)  # (N, )
        ind_objectives = self._normalize(self._extract_objectives(front), fmin, fmax)  # (N, M)

        selected = []
        is_alive = np.ones(front_size, dtype=bool)
        while len(selected) < needed and is_alive.any():
            candidate_indices = np.where(is_alive)[0]
            # run selection based on preference rank
            best_rank = pref_rank[candidate_indices].min()
            best_ranked_indices = candidate_indices[pref_rank[candidate_indices] == best_rank]
            selected_idx = self.rng.choice(best_ranked_indices)
            selected.append(front[selected_idx])
            is_alive[selected_idx] = False
            # clear neighbors in normalized L1 within epsilon
            if self.epsilon > 0:
                diffs_l1 = np.abs(ind_objectives[candidate_indices] - ind_objectives[selected_idx]).sum(axis=1)
                kill = candidate_indices[diffs_l1 <= self.epsilon]
                is_alive[kill] = False
        # if we ran out due to clearing, fill randomly from remaining (typical safeguard)
        if len(selected) < needed:
            rest = [ind for j, ind in enumerate(front) if not is_alive[j]]
            self.rng.shuffle(rest)
            selected.extend(rest[: needed - len(selected)])
        return selected

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
        fronts = self.fast_nondominated_sort(population.extract(list(indices)))
        if len(fronts[0]) == 1:
            return fronts[0][0]
        if self.reference_points is not None:
            # normalization scope for parent selection: use group itself (stable & cheap)
            fmin, fmax = self._minmax_on(list(self.population.values()))
            pref_rank = self._preference_rank(fronts[0], fmin, fmax)  # lower is better
            # choose min pref (break ties randomly)
            best = np.where(pref_rank == pref_rank.min())[0]
            return fronts[0][int(self.rng.choice(best))]
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
        return polynomial_mutation(ind, self.bounds, self.rng, eta=eta, pm=pm)

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
            if len(next_pop) + len(front) <= self.population_size:
                next_pop.extend(front)
                continue
            if self.reference_points is not None:
                # preference-based selection
                need = self.population_size - len(next_pop)
                fmin, fmax = self._minmax_on(list(self.population.values()))
                chosen = self._epsilon_clearing_select(front, need, fmin, fmax)
                next_pop.extend(chosen)
            else:
                # crowding-distance-based selection
                crowding = self.crowding_distance(front)
                sorted_front = [
                    ind
                    for _, ind in sorted(zip(crowding, front, strict=True), key=lambda zipped: zipped[0], reverse=True)
                ]
                next_pop.extend(sorted_front[: self.population_size - len(next_pop)])
            break
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
def build_nsga2(  # noqa: PLR0913
    dim: int,
    num_obj: int,
    population_size: int | None = None,
    num_children: int | None = None,
    bounds: tuple[float, float] | list[tuple[float, float]] | None = None,
    seed: int | None = None,
    mean: np.ndarray | None = None,
    reference_points: np.ndarray | None = None,
    reference_weights: np.ndarray | None = None,
    epsilon: float = 1e-3,
    init_distribution_eta: float = 2.0,
) -> OptimizerInterface:
    """
    NSGA-II multi-objective evolutionary optimization algorithm class including constraint handling.
    Inherits from MOOptimizerBase and OptimizerInterface.

    Args:
        dim (int): optimization problem dimension (Individual size).
            When Using pyemsol_shape_evaluator, this will be automatically set by EMSOptimizer.
        num_obj (int): Number of objectives.
            When Using pyemsol_shape_evaluator, this will be automatically set by EMSOptimizer.
        population_size (int | None, optional): Population size. If None, defaults to num_obj * 100.
        num_children (int | None, optional): Number of offspring. If None, defaults to population_size.
        bounds (tuple[float, float] | list[tuple[float, float]] | None , optional): Range for each variable
        seed (int | None, optional): Random seed for reproducibility.
        mean (np.ndarray | None): Initial center solution. Short vectors are padded with zeros.
            Raises an error if the vector is longer than dim.
        reference_points (np.ndarray (K, M) | None): reference points in objective space (minimization).
        reference_weights (np.ndarray (K, M) | None): Optional positive weights per reference point; defaults to ones.
        epsilon (float): Clearing width in normalized L1 distance on objective space.
        init_distribution_eta (float): Polynomial mutation eta used when initializing from mean.
    """
    return NSGA2(
        dim=dim,
        num_obj=num_obj,
        population_size=population_size,
        num_children=num_children,
        bounds=bounds,
        seed=seed,
        mean=mean,
        reference_points=reference_points,
        reference_weights=reference_weights,
        epsilon=epsilon,
        init_distribution_eta=init_distribution_eta,
    )
