"""
cmaes.py
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

import numpy as np
from cmaes import CMA

from core.optimizer.examples.helpers.decomposition import (
    Scalarizer,
    generate_scalarizer,
    initialize_reference_and_nadir,
    update_reference_and_nadir,
)
from core.optimizer.examples.helpers.variation import normalize_mean
from emsopt_engine.individual import Individual, Population
from emsopt_engine.interface.optimizer_interface import OptimizerInterface
from emsopt_engine.interface.so_optimizer_base import SOOptimizerBase
from emsopt_engine.registry import optimizer


class CMAES(SOOptimizerBase, OptimizerInterface):
    EPS = 1e-10
    _INV_CLIP = 1e-12  # avoid logit inf

    def __init__(
        self,
        dim: int,
        num_obj: int = 1,
        mean: np.ndarray | None = None,
        sigma: float = 1.0,
        bounds: tuple[float, float] | list[tuple[float, float]] | None = None,
        seed: int | None = None,
        population_size: int | None = None,
        scalarizer: Scalarizer | None = None,
        scalarizer_weights: list[float] | np.ndarray | None = None,
    ) -> None:
        if not isinstance(dim, int) or dim <= 0:
            msg = "dim must be a positive integer"
            raise ValueError(msg)
        if not isinstance(num_obj, int) or num_obj <= 0:
            msg = "num_obj must be a positive integer"
            raise ValueError(msg)
        mean = normalize_mean(mean, dim)
        if bounds is not None:
            if np.asarray(bounds).ndim == 1:
                bounds = [bounds] * dim
            elif (np.asarray(bounds).ndim == 2) and (len(bounds) <= dim):  # noqa: PLR2004
                # fill bounds by (-1, 1)
                for _ in range(dim - len(bounds)):
                    bounds.append((-1, 1))
            else:
                msg = "bounds must be a tuple or list of tuples with length equal to or smaller than dim."
                raise ValueError(msg)
        self.bounds = np.asarray(bounds) if bounds is not None else None
        self._use_transform = self.bounds is not None  # use logistic transform if bounds are given
        # z-space mean: if user gave mean in x-space and bounds exist, map to z via inverse-logit
        if mean is None:
            z_mean = np.zeros(dim)
        else:
            z_mean = (
                self._x_to_z(np.asarray(mean, dtype=float)) if self._use_transform else np.asarray(mean, dtype=float)
            )
        super().__init__(population_size, seed)  # Call SOOptimizerBase's __init__ according to Python's MRO
        cma_bounds = None if self._use_transform else self.bounds
        self.cma = CMA(mean=z_mean, sigma=sigma, bounds=cma_bounds, seed=seed, population_size=population_size)
        if self.population_size is None:
            self.population_size = self.cma.population_size
        self.num_obj = num_obj
        self.scalarizer = scalarizer
        self.scalarizer_weights = scalarizer_weights
        self._z_ref: list[float] | None = None
        self._z_nadir: list[float] | None = None

    # --------- logistic transform helpers (per-dimension [a_i, b_i]) ----------
    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # numerically stable sigmoid
        out = np.empty_like(z, dtype=float)
        pos = z >= 0
        neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[neg])
        out[neg] = ez / (1.0 + ez)
        return out

    def _z_to_x(self, z: np.ndarray) -> np.ndarray:
        """Map z∈R^d to x within bounds via per-dim affine(logistic(z))."""
        if not self._use_transform:
            return z.copy()
        u = self._sigmoid(np.asarray(z, dtype=float))
        a = self.bounds[:, 0]
        b = self.bounds[:, 1]
        return a + (b - a) * u

    def _x_to_z(self, x: np.ndarray) -> np.ndarray:
        """Inverse map: x in [a,b] -> z in R via per-dim logit((x-a)/(b-a))."""
        if not self._use_transform:
            return x.copy()
        x = np.asarray(x, dtype=float)
        a = self.bounds[:, 0]
        b = self.bounds[:, 1]
        u = (x - a) / np.maximum(b - a, self.EPS)
        # clip to avoid logit inf
        u = np.clip(u, self._INV_CLIP, 1.0 - self._INV_CLIP)
        return np.log(u / (1.0 - u))

    # -------------------------------------------------------------------------

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

    def setup_population(self, evaluated_population: Population) -> None:  # noqa: ARG002 ; for interface compatibility
        """Not needed for CMAES"""
        return

    def get_candidates(self) -> Population:
        """
        Sample a population from the current distribution
        Returns:
            Population: Population of individuals
        Raises:
            Exception: If an error occurs during population generation
        """
        self.population = Population()
        # Generate population using cmaes ask
        pop_size = self.cma.population_size
        for idx in range(pop_size):
            z = np.asarray(self.cma.ask(), dtype=float)  # latent sample in z-space
            x = self._z_to_x(z)  # mapped to box domain (or identity if no bounds)
            ind = Individual(list(x))
            # keep latent z for tell(); safe to stash as ad-hoc attribute
            ind.z = z
            self.population[idx] = ind
        return self.population

    def proceed_to_next_iteration(self, evaluated_candidates: Population) -> None:
        """
        Update distribution parameters based on evaluated_candidates.
        Uses latent z (if present) for CMA tell(); evaluation fitness is unchanged.
        """
        self.population = evaluated_candidates
        if self.scalarizer is not None:
            if self.scalarizer_weights is None:
                msg = "scalarizer_weights must be provided when scalarizer is enabled"
                raise ValueError(msg)
            self._update_reference_and_nadir(self.population)
            for ind in self.population.values():
                ind.metrics.fitness = self.scalarizer(ind.metrics.objectives, self.scalarizer_weights)
        self.update_best_individual()

        # collect fitnesses; use latent z if available
        solutions_fitnesses: list[tuple[np.ndarray, float]] = []

        worst_fitness = max(ind.metrics.fitness for ind in self.population.values())
        for ind in self.population.values():
            # latent variable to update CMA in the same space as ask()
            z = getattr(ind, "z", np.asarray(ind.solution, dtype=float))
            if ind.metrics.constraint_violation > self.EPS:
                fit = worst_fitness + ind.metrics.constraint_violation
            else:
                fit = ind.metrics.fitness
            solutions_fitnesses.append((z, fit))

        self.cma.tell(solutions_fitnesses)


@optimizer("cmaes")
def build_cmaes(
    dim: int,
    num_obj: int,
    mean: np.ndarray | None = None,
    sigma: float = 1.0,
    bounds: tuple[float, float] | list[tuple[float, float]] | None = None,
    seed: int | None = None,
    population_size: int | None = None,
    scalarizer_type: str | None = None,
    scalarizer_weights: list[float] | np.ndarray | None = None,
) -> OptimizerInterface:
    """
    Wrapper class for the CMA-ES algorithm.
    Manages populations/individuals using the cmaes library.
    Generates populations with get_population and updates distribution parameters with proceed_to_next_iteration.

    Args:
        dim (int): optimization problem dimension (Individual size).
            When Using pyemsol_shape_evaluator, this will be automatically set by EMSOptimizer.
        num_obj (int): Number of objectives.
            When Using pyemsol_shape_evaluator, this will be automatically set by EMSOptimizer.
        mean (np.ndarray, optional): Initial mean vector. Short vectors are padded with zeros.
            Raises an error if the vector is longer than dim.
        sigma (float, optional): Initial standard deviation. Defaults to 1.0.
        bounds (tuple[float, float] | list[tuple[float, float]], optional): bounds of each variable
        seed (int | None, optional): Random seed for reproducibility.
        population_size (int | None, optional): Population size.
            If None, proper value is set automatically by cmaes library.
        scalarizer_type (str | None): scalarizer type that scalarize multiple objectives (metrics.objectives).
            If None, metrics.fitness is directly used as single objective. Defaults to None.
        scalarizer_weights (list[float] | np.ndarray | None): Weight coefficients used by the scalarizer.
            Required when scalarizer_type is specified.
    Raises:
        ValueError: If individual_size is not positive or mean is longer than dim
    """
    scalarizer = generate_scalarizer(scalarizer_type) if scalarizer_type is not None else None
    return CMAES(
        dim=dim,
        num_obj=num_obj,
        mean=mean,
        sigma=sigma,
        bounds=bounds,
        seed=seed,
        population_size=population_size,
        scalarizer=scalarizer,
        scalarizer_weights=scalarizer_weights,
    )
