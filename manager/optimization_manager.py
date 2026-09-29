"""
optimization_manager.py
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

import datetime
import gc
import queue
import shutil
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from logging import getLogger
from multiprocessing import Event, Process, cpu_count, get_context
from pathlib import Path

import yaml
from injector import inject

from emsopt_engine.configs.base import BaseConfig
from emsopt_engine.configs.optimization import OptimizationConfig
from emsopt_engine.individual import Population
from emsopt_engine.interface.evaluator_interface import EvaluatorInterface
from emsopt_engine.interface.mo_optimizer_base import MOOptimizerBase
from emsopt_engine.interface.optimizer_interface import OptimizerInterface
from emsopt_engine.interface.so_optimizer_base import SOOptimizerBase
from manager.optimization_output_handler import (
    MOHandler,
    OptimizationOutputHandlerInterface,
    SOHandler,
    load_individuals,
)
from manager.optimization_visualizer import OptimizationVisualizer, VisualizerPayload

logger = getLogger(__name__)


@dataclass
class OptimizationSummary:
    """Optimization summary data"""

    num_iteration: int
    num_candidates_per_iteration: int
    num_parallel_process: int
    total_computation_time: float


class OptimizationManager:
    """
    Manager class for overall optimization.
    Receives optimizer (optimization algorithm) and evaluator (evaluation system),
    executes the optimization loop with run_optimization.
    """

    @inject  # dependency injection by Injector
    def __init__(
        self, optimizer: OptimizerInterface, evaluator: EvaluatorInterface, config: OptimizationConfig
    ) -> None:
        """
        Initialize OptimizationManager
        Args:
            optimizer (OptimizerInterface): Optimization algorithm
            evaluator (EvaluationInterface): Evaluation system
            config (OptimizationConfig): Optimization config
            analyzer (OptimizationAnalyzer): Optimization Analyzer
        Raises:
            TypeError: If argument types are invalid
        """
        # members
        if not isinstance(optimizer, OptimizerInterface):
            msg = "optimizer must be instance of OptimizerInterface"
            raise TypeError(msg)
        if not isinstance(evaluator, EvaluatorInterface):
            msg = "evaluator must be instance of EvaluatorInterface"
            raise TypeError(msg)
        if not isinstance(config, OptimizationConfig):
            msg = "config must be instance of OptimizationConfig"
            raise TypeError(msg)
        self.optimizer: OptimizerInterface = optimizer
        self.evaluator: EvaluatorInterface = evaluator
        self.config: OptimizationConfig = config

        # gui and related
        self.visualizer: OptimizationVisualizer = OptimizationVisualizer(self.config.output_dir)
        self.visualizer_p: Process | None = None
        self.progress_q: object | None = None
        self.stop_evt: Event = None
        self.pause_evt: Event = None
        self.time_iterations: list[datetime.timedelta] = []

        # output handler
        self.output_handler: OptimizationOutputHandlerInterface | None = None
        if isinstance(self.optimizer, SOOptimizerBase) and isinstance(self.optimizer, MOOptimizerBase):
            msg = "optimizer cannot be both SOOptimizerBase and MOOptimizerBase"
            raise TypeError(msg)
        if isinstance(self.optimizer, SOOptimizerBase):
            self.output_handler = SOHandler()
        elif isinstance(self.optimizer, MOOptimizerBase):
            self.output_handler = MOHandler()
        else:
            logger.warning("Unsupported optimizer type for handler. Most of output functions will be disabled.")
        self._init_output_dir()

    @property
    def elapsed(self) -> datetime.timedelta:
        """Elapsed time"""
        return sum(self.time_iterations, datetime.timedelta(0))

    @property
    def eta(self) -> datetime.timedelta:
        """Estimated time remaining"""
        if len(self.time_iterations) == 0:
            return datetime.timedelta(0)
        avg_time = self.elapsed / len(self.time_iterations)
        remaining_iters = self.config.num_iteration - len(self.time_iterations)
        return avg_time * remaining_iters

    def _init_output_dir(self) -> None:
        """create output directory based on config"""
        output_dir = Path(self.config.output_dir)
        if output_dir.exists():
            logger.warning("output dir already exists. Files in the dir will be overwritten!")
        else:
            output_dir.mkdir()

    def run_optimization(self, summary_dst_dir: str) -> None:
        """
        Execute optimization

        Args:
            summary_dst_dir (str): destination directory to save summary

        Raises:
            Exception: If an error occurs during processing
        """
        breaked = False
        try:
            # pre process
            self._setup_optimizer()
            if self.config.enable_progress_gui:
                self._setup_visualizer()

            # optimization loop
            for i_iter in range(self.config.num_iteration):
                if self._check_events():
                    breaked = True
                    break
                start = datetime.datetime.now(tz=datetime.UTC)
                evaluated_candidates = self._run_optimization_body()
                self.time_iterations.append(datetime.datetime.now(tz=datetime.UTC) - start)
                self._post_iteration(i_iter, evaluated_candidates)

            # post process
            resultant_iter = (i_iter + 1) if breaked else self.config.num_iteration
            summary = OptimizationSummary(
                num_iteration=resultant_iter,
                num_candidates_per_iteration=len(evaluated_candidates),
                num_parallel_process=cpu_count()
                if self.config.num_processes is None
                else min(self.config.num_processes, cpu_count()),
                total_computation_time=self.elapsed.total_seconds(),
            )
            self._after_optimization(summary, summary_dst_dir)

        finally:
            self._finalize_visualizer()

    def check_optimization_result(self, summary_dst_dir: str) -> None:
        """
        Check performed optimization result

        Args:
            summary_dst_dir (str): destination directory to load summary
        """
        try:
            self._setup_visualizer()
            # load result
            base_dir = Path(summary_dst_dir) / BaseConfig.RESULTANT_DIRNAME.value
            filepath = base_dir / self.config.output_control["best_individual"].filename_base
            individuals = load_individuals(str(filepath.with_suffix(".yaml")))

            # replace outcome_filepath
            for ind in individuals:
                if ind.outcome_filepath:
                    ind.outcome_filepath = str(base_dir / Path(ind.outcome_filepath).name)

            # Send payload and wait for GUI to close.
            self._send_payload(VisualizerPayload("update", individuals=deepcopy(individuals)))
            if self.visualizer_p and self.visualizer_p.is_alive():
                logger.info("To end process, please close GUI, or press Ctrl + C on CUI to terminate.")
                self.visualizer_p.join()

        except Exception as e:
            msg = f"Error during loading result: {e}. Return."
            logger.exception(msg)
            raise Exception(msg) from e

        finally:
            self._finalize_visualizer()

    def _setup_optimizer(self) -> None:
        """Set up the optimizer (for population-based optimizers)"""
        population = self.optimizer.get_population()
        if population:
            logger.info("Running initial evaluation of population...")
            if self.config.enable_parallelization:
                evaluated_population = self.evaluator.evaluate_parallel(population, self.config.num_processes)
            else:
                evaluated_population = self.evaluator.evaluate(population)
            self.optimizer.setup_population(evaluated_population)
            self._post_iteration(-1, evaluated_population)

    def _setup_visualizer(self) -> None:
        """launch GUI with proper mode for optimizer"""
        if isinstance(self.optimizer, SOOptimizerBase):
            mode = "SO"
        elif isinstance(self.optimizer, MOOptimizerBase):
            mode = "MO"
        else:
            logger.warning("Unsupported optimizer type for visualization.")
            return
        ctx = get_context("spawn")  # cross-platform stable
        self.progress_q = ctx.Queue(maxsize=self.config.num_iteration)
        self.stop_evt = ctx.Event()
        self.pause_evt = ctx.Event()
        self.visualizer_p = ctx.Process(
            target=self.visualizer.run,
            args=(mode, self.evaluator.get_num_objectives(), self.progress_q, self.stop_evt, self.pause_evt),
            daemon=True,
        )
        self.visualizer_p.start()

    def _check_events(self) -> bool:
        """Check stop/pause events from gui. Returns True if should break loop."""
        if self.stop_evt is not None and self.stop_evt.is_set():
            return True
        if self.pause_evt is not None:
            while self.pause_evt.is_set():
                if self.stop_evt is not None and self.stop_evt.is_set():
                    return True
                time.sleep(0.05)
        return False

    def _run_optimization_body(self) -> Population:
        """Main optimization body: candidate generation, evaluation, update."""
        candidates = self.optimizer.get_candidates()
        if self.config.enable_parallelization:
            evaluated_candidates = self.evaluator.evaluate_parallel(candidates, self.config.num_processes)
        else:
            evaluated_candidates = self.evaluator.evaluate(candidates)
        self.optimizer.proceed_to_next_iteration(evaluated_candidates)
        del candidates
        gc.collect()
        return evaluated_candidates

    def _post_iteration(self, i_iter: int, evaluated_candidates: Population) -> None:
        """Post-iteration: output progress, send payload."""
        if self.output_handler and ((i_iter + 1) % self.config.output_interval == 0):
            self._output_progress(evaluated_candidates, i_iter)
            if self.visualizer_p and self.visualizer_p.is_alive() and self.progress_q:
                if isinstance(self.optimizer, SOOptimizerBase):
                    individuals = self.optimizer.best_history
                elif isinstance(self.optimizer, MOOptimizerBase):
                    individuals = list(self.optimizer.pareto.values())
                else:
                    logger.warning("Unsupported optimizer type for visualization.")
                    return
                self._send_payload(VisualizerPayload("update", self.elapsed, self.eta, deepcopy(individuals)))
        logger.info(f"Iteration {i_iter + 1} finished")

    def _after_optimization(self, summary: OptimizationSummary, summary_dst_dir: str) -> None:
        """After optimization pocess"""
        # Save summary
        if Path(summary_dst_dir).exists():
            logger.warning("Optimization summary of this project already exists. Overwriting...")
            shutil.rmtree(summary_dst_dir)
        Path(summary_dst_dir).mkdir()
        filepath = Path(summary_dst_dir) / BaseConfig.SUMMARY_FILENAME.value
        with filepath.open(mode="w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(summary), f)
        # copy output dir
        self.output_handler.save_latest_summary_dir(str(Path(summary_dst_dir) / BaseConfig.RESULTANT_DIRNAME.value))
        # Wait for GUI to close.
        if self.visualizer_p and self.visualizer_p.is_alive():
            logger.info(
                "Optimization finished. To end process, please close GUI, or press Ctrl + C on CUI to terminate."
            )
            self.visualizer_p.join()

    def _finalize_visualizer(self) -> None:
        """Finalize: clean up visualizer process and resources."""
        logger.info("Gracefully closing... (Press Ctrl + C to terminate)")
        if self.visualizer_p and self.visualizer_p.is_alive():
            if self.progress_q:
                self._send_payload(VisualizerPayload(type="done"))
            self.visualizer_p.join()
        if self.progress_q:
            self.progress_q.close()
            self.progress_q.join_thread()

    def _send_payload(self, payload: VisualizerPayload) -> None:
        """send payload to visualizer"""
        try:
            self.progress_q.put_nowait(payload)
        except queue.Full:
            msg = "Visualization data queue is full. skip sending data."
            logger.warning(msg)

    def _output_progress(self, evaluated_candidates: Population, i_iter: int) -> None:
        """Output progress based on optimizer (should be subclass of SOOptimizerBase or MOOptimizerBase)
        Args:
            evaluated_candidates (Population): evaluated candidates in current iter
            i_iter (int): current iteration
        """
        output_dir = Path(self.config.output_dir)
        try:
            for key, outconf in self.config.output_control.items():
                if (not outconf.enabled) or ((i_iter + 1) % outconf.output_interval != 0):
                    continue
                match key:
                    case "best_individual":
                        self.output_handler.output_best_individual(self.optimizer, output_dir, i_iter, outconf)
                    case "candidate":
                        self.output_handler.output_candidate(evaluated_candidates, output_dir, i_iter, outconf)
                    case "candidate_plot":
                        self.output_handler.output_candidate_plot(evaluated_candidates, output_dir, i_iter, outconf)
        except Exception as e:
            logger.warning("Output progress failed. Continue optimization run.")
            logger.warning("Error message: %s", str(e))
