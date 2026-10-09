"""
sa_nsga2.py
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
from pathlib import Path
from typing import Literal

import numpy as np

from core.optimizer.examples.nsga2 import NSGA2
from core.optimizer.surrogate_models.factory import resolve_surrogate_model
from core.optimizer.surrogate_models.protocol import SurrogateProtocol
from core.optimizer.surrogate_utils import (
    SurrogateTargetAdapter,
    mark_surrogate_prediction,
    mark_true_evaluation,
)
from emsopt_engine.individual import Individual, Population
from emsopt_engine.interface.optimizer_interface import OptimizerInterface
from emsopt_engine.registry import optimizer
from utils.individuals_io import SURROGATE_INFO_ATTRIBUTE, dump_individuals_csv, load_individuals_csv


class SANSGA2(NSGA2):
    """Surrogate-assisted NSGA-II with shared target-adaptation utilities.

    The optimizer itself focuses on NSGA-II specific flow. The choice of
    surrogate target space (objectives vs. label_values) is delegated to
    `SurrogateTargetAdapter`, and bookkeeping of prediction/true-evaluation
    traces is delegated to the shared tracking helpers.
    """

    EPS = 1e-10

    def __init__(  # noqa: PLR0913
        self,
        dim: int,
        num_obj: int,
        surrogate: SurrogateProtocol,
        population_size: int | None = None,
        num_children: int | None = None,
        bounds: tuple[float, float] | list[tuple[float, float]] | None = None,
        seed: int | None = None,
        mean: np.ndarray | None = None,
        reference_points: np.ndarray | None = None,
        reference_weights: np.ndarray | None = None,
        epsilon: float = 1e-3,
        init_distribution_eta: float = 2.0,
        *,
        func_manager: object | None = None,
        enable_adaptive_training: bool = False,
        surrogate_mode: Literal["full", "full_loop", "assist", "assist_strict"] = "assist_strict",
        eval_init_population_truly: bool = True,
    ) -> None:
        super().__init__(
            dim,
            num_obj,
            population_size,
            num_children,
            bounds,
            seed,
            mean,
            reference_points,
            reference_weights,
            epsilon,
            init_distribution_eta,
        )
        self.offspring: Population | None = None
        self.surrogate = surrogate
        self.surrogate_mode = surrogate_mode
        self.enable_adaptive_training = enable_adaptive_training
        self.evaluation_rank = 1
        self.rest_evaluation_ratio = 0.1
        self.adaptive_training_data: list[Individual] = []
        self.num_inner_loop_iteration = 100
        self.inner_loop_count = 0
        self.target_adapter = SurrogateTargetAdapter(func_manager)
        self.func_manager = func_manager
        self.target_label_names = list(self.target_adapter.target_label_names)
        self.uses_label_targets = self.target_adapter.uses_label_targets
        if not eval_init_population_truly:
            # Delay initial true evaluation and start from surrogate prediction only.
            self.population = None
            self.mean = mean

    def _init_population_from_mean(self) -> None:
        """Create the first population around the restart/seed mean."""
        self._init_population(self.mean)

    def _apply_surrogate_prediction(self, individual: Individual, predicted_targets: np.ndarray | list[float]) -> None:
        """Reflect surrogate outputs into one individual and store prediction traces."""
        predicted_objectives, predicted_label_values = self.target_adapter.apply_prediction(
            individual, predicted_targets
        )
        mark_surrogate_prediction(
            individual,
            predicted_objectives=predicted_objectives,
            predicted_label_values=predicted_label_values,
        )

    def setup_population(self, evaluated_population: Population) -> None:
        """Accept an externally evaluated population and optionally retrain the surrogate."""
        super().setup_population(evaluated_population)
        if self.enable_adaptive_training:
            self.store_training_data(self.population)
            self.adaptive_train()

    def get_candidates(self) -> Population:
        """Generate candidate individuals according to the selected surrogate mode."""
        if self.population is None:
            self._init_population_from_mean()
            pred, _ = self.surrogate.predict(np.asarray(self.population.to_design_matrix(), dtype=float))
            for ind, obj in zip(self.population.values(), pred, strict=True):
                self._apply_surrogate_prediction(ind, obj)
        if not self.surrogate.is_ready():
            self.offspring = super().get_candidates()
            return deepcopy(self.offspring)
        if self.surrogate_mode == "full":
            self.offspring = self._get_offspring_with_prediction()
            return Population()
        if self.surrogate_mode == "full_loop":
            return self._inner_loop()
        if self.surrogate_mode in ["assist", "assist_strict"]:
            self.offspring = self._get_offspring_with_prediction()
            return self._ranking_based_sampling()
        msg = f"Unknown surrogate mode {self.surrogate_mode}"
        raise ValueError(msg)

    def proceed_to_next_iteration(self, evaluated_candidates: Population) -> None:
        """Merge true evaluations back into offspring, then run the normal NSGA-II update."""
        for idx, individual in evaluated_candidates.items():
            predicted_value = None
            if self.offspring is not None and idx in self.offspring:
                surrogate_info = getattr(self.offspring[idx], SURROGATE_INFO_ATTRIBUTE, {})
                if isinstance(surrogate_info, dict) and "predicted_value" in surrogate_info:
                    predicted_value = surrogate_info["predicted_value"]
            mark_true_evaluation(
                individual,
                predicted_value=predicted_value,
                true_label_values=self.target_adapter.true_label_values_for_tracking(individual),
            )
        self.offspring.update(evaluated_candidates)
        super().proceed_to_next_iteration(self.offspring)
        if self.enable_adaptive_training:
            self.store_training_data(evaluated_candidates)
            self.adaptive_train()

    def store_training_data(self, evaluated_candidates: Population) -> None:
        """Append newly true-evaluated individuals to the adaptive training buffer."""
        self.adaptive_training_data.extend(
            individual for individual in evaluated_candidates.values() if individual.record_info.status != "failure"
        )

    def adaptive_train(self) -> None:
        """Train/update the surrogate using the adapter-selected target space."""
        x_data = [np.asarray(ind.solution, dtype=float) for ind in self.adaptive_training_data]
        y_data = [self.target_adapter.extract_training_targets(ind) for ind in self.adaptive_training_data]
        if not self.surrogate.is_ready():
            self.surrogate.fit(np.asarray(x_data, dtype=float), np.asarray(y_data, dtype=float))
        else:
            self.surrogate.update(np.asarray(x_data, dtype=float), np.asarray(y_data, dtype=float))

    def _get_offspring_with_prediction(self) -> Population:
        """Create offspring and attach surrogate predictions before selection/filtering."""
        offspring = super().get_candidates()
        pred, _ = self.surrogate.predict(np.asarray(offspring.to_design_matrix(), dtype=float))
        for ind, obj in zip(offspring.values(), pred, strict=True):
            self._apply_surrogate_prediction(ind, obj)
        return offspring

    def _inner_loop(self) -> Population:
        """Run repeated surrogate-only generations, then return the final population for true evaluation."""
        original_population = deepcopy(self.population)
        original_archive = deepcopy(self.archive)
        original_pareto = deepcopy(self.pareto)
        for _ in range(self.num_inner_loop_iteration):
            offspring = self._get_offspring_with_prediction()
            super().proceed_to_next_iteration(offspring)
        self.offspring = deepcopy(self.population)
        dump_individuals_csv(
            list(self.offspring.values()), f"./last_population_of_inner_loop_{self.inner_loop_count + 1}.csv"
        )
        self.inner_loop_count += 1
        self.population = original_population
        self.archive = original_archive
        self.pareto = original_pareto
        return deepcopy(self.offspring)

    def _ranking_based_sampling(self) -> Population:
        """Select which surrogate-evaluated offspring will be sent to true evaluation."""
        offspring = deepcopy(self.offspring)
        fronts = self.fast_nondominated_sort(offspring)
        if self.evaluation_rank > len(fronts):
            offspring_for_evaluation = deepcopy(offspring)
        else:
            offspring_for_evaluation = Population(
                dict(enumerate([ind for front in fronts[: self.evaluation_rank] for ind in front]))
            )
            rest = Population(
                dict(
                    enumerate(
                        [ind for front in fronts[self.evaluation_rank :] for ind in front],
                        start=len(offspring_for_evaluation),
                    )
                )
            )
            keys = self.rng.choice(
                list(rest.keys()),
                size=int(len(rest) * self.rest_evaluation_ratio),
                replace=False,
            )
            offspring_for_evaluation.update({k: rest[k] for k in keys})
        if self.surrogate_mode == "assist_strict":
            # In strict mode, only truly evaluated offspring may survive this generation.
            self.offspring = offspring_for_evaluation
        with Path("./evaluation_ratio.csv").open("a") as f:
            print(len(offspring_for_evaluation), len(offspring_for_evaluation) / self.n_children, sep=",", file=f)
        return offspring_for_evaluation


@optimizer("sa_nsga2")
def build_sa_nsga2(  # noqa: PLR0913
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
    surrogate_model: str = "mlp",
    data_csv_path: str | None = None,
    func_manager: object | None = None,
    *,
    enable_adaptive_training: bool = True,
    surrogate_mode: Literal["full", "full_loop", "assist", "assist_strict"] = "assist_strict",
    eval_init_population_truly: bool = True,
) -> OptimizerInterface:
    """
    NSGA-II multi-objective evolutionary optimization algorithm class including constraint handling.
    Inherits from MOOptimizerBase and OptimizerInterface.
    Assisted by surrogate model to reduce true evaluations.

    Additional args:
        surrogate_model (str): built-in surrogate name.
        data_csv_path (int | None): data csv path for training surrogate model.
            If None, initial training will be skipped.
        func_manager (object | None): EMSOptimizer's function manager which will be automatically injected.
        enable_adaptive_training (bool): enable adaptive training. Defaults to False.
        surrogate_mode (Literal["full", "assist", "assist_strict"]): surrogate assist mode. Defaults to "assist".
            If "full", all evaluations will be replaced with surrogate model.
            If "full_loop", "full" is repeated until max iteration reached.
                Each results of "full" are passed to evaluator for true evaluation.
            If "assist", evaluations will be partially replaced and partially truly evaluated.
            If "assist_strict", based on "assist", individuals truly evaluated will be definately considered.
        eval_init_population_truly (bool): Whether init population is truly evaluated or not. Defaults to True.
    """
    if data_csv_path is None and not enable_adaptive_training:
        msg = (
            "'data_csv_path' is None and 'enable_adaptive_training' is False, so cannot perform any training. "
            "Please set at least one of them."
        )
        raise ValueError(msg)

    target_adapter = SurrogateTargetAdapter(func_manager)
    surrogate_num_targets = target_adapter.num_targets or num_obj
    surrogate_impl = resolve_surrogate_model(
        surrogate_model=surrogate_model,
        dim=dim,
        num_obj=surrogate_num_targets,
        seed=seed,
    )

    model = SANSGA2(
        dim=dim,
        surrogate=surrogate_impl,
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
        func_manager=func_manager,
        enable_adaptive_training=enable_adaptive_training,
        surrogate_mode=surrogate_mode,
        eval_init_population_truly=eval_init_population_truly,
    )

    if data_csv_path is not None:
        # Initial CSV data participates in the same adaptive-training pipeline as online true evaluations.
        model.store_training_data(Population(dict(enumerate(load_individuals_csv(data_csv_path)))))
        if not eval_init_population_truly:
            model.adaptive_train()

    return model
