"""
optimization_manager.py
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

import datetime
import gc
import math
import queue
import shutil
import threading
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from logging import getLogger
from multiprocessing import Process, cpu_count, get_context
from numbers import Real
from pathlib import Path

import yaml
from injector import inject
from scipy.stats import qmc

from emsopt_engine.configs.base import BaseConfig
from emsopt_engine.configs.optimization import OptimizationConfig, ResponseSurfaceConfig
from emsopt_engine.individual import Population
from emsopt_engine.interface.evaluator_interface import EvaluatorInterface
from emsopt_engine.interface.mo_optimizer_base import MOOptimizerBase
from emsopt_engine.interface.optimizer_interface import OptimizerInterface
from emsopt_engine.interface.so_optimizer_base import SOOptimizerBase
from manager.optimization_output_handler import (
    MOHandler,
    OptimizationOutputHandlerInterface,
    SOHandler,
)
from manager.optimization_visualizer import OptimizationVisualizer, VisualizerPayload
from manager.response_surface import ResponseSurfaceRequest, ResponseSurfaceSession, build_response_surface_session
from manager.restart_selection import RestartSelection
from utils.individuals_io import load_individuals, resolve_outcome_filepath

logger = getLogger(__name__)
BOUNDS_PAIR_LENGTH = 2


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
        self.stop_evt = None  # multiprocessing.Event
        self.pause_evt = None  # multiprocessing.Event
        self.restart_q: object | None = None
        self.response_surface_request_q: object | None = None
        self.restart_selection: RestartSelection | None = None
        self.project_name: str | None = None
        self.study_name: str | None = None
        self.restart_target_study_name: str | None = None
        self.control_provider: object | None = None
        self.stopped_by_request = False
        self.current_summary_dir: str | None = None
        self.time_iterations: list[datetime.timedelta] = []
        self.response_surface_session: ResponseSurfaceSession | None = None
        self._response_surface_worker_thread: threading.Thread | None = None
        self._response_surface_worker_stop = threading.Event()

        # output handler
        self.output_handler: OptimizationOutputHandlerInterface | None = None
        if isinstance(self.optimizer, SOOptimizerBase) and isinstance(self.optimizer, MOOptimizerBase):
            msg = "optimizer cannot be both SOOptimizerBase and MOOptimizerBase"
            raise TypeError(msg)
        material_color_map = self._resolve_material_color_map()
        image_export = self._resolve_image_export()
        if isinstance(self.optimizer, SOOptimizerBase):
            self.output_handler = SOHandler(material_color_map=material_color_map, image_export=image_export)
        elif isinstance(self.optimizer, MOOptimizerBase):
            self.output_handler = MOHandler(material_color_map=material_color_map, image_export=image_export)
        else:
            logger.warning("Unsupported optimizer type for handler. Most of output functions will be disabled.")
        self._init_output_dir()

    def _resolve_machine_config(self) -> object | None:
        return getattr(self.evaluator, "machine_conig", None) or getattr(self.evaluator, "machine_config", None)

    def _resolve_material_color_map(self) -> dict[int, str]:
        machine_config = self._resolve_machine_config()
        material_color_map = getattr(machine_config, "material_color_map", None)
        if isinstance(material_color_map, dict):
            return dict(material_color_map)
        return {}

    def _resolve_image_export(self) -> dict[str, object]:
        machine_config = self._resolve_machine_config()
        image_export = getattr(machine_config, "image_export", None)
        if image_export is None:
            return {}
        resolution = getattr(image_export, "resolution", None)
        view_mode = getattr(image_export, "view_mode", None)
        if not isinstance(resolution, int) or not isinstance(view_mode, str):
            return {}
        export_settings = {
            "resolution": resolution,
            "view_mode": view_mode,
            "sym_deg": float(getattr(machine_config, "sym_deg", 0.0)),
            "num_rotate": int(getattr(machine_config, "num_rotate", 0)),
        }
        image_target_region = getattr(image_export, "image_target_region", None)
        if image_target_region is not None:
            export_settings["image_target_region"] = tuple(tuple(axis) for axis in image_target_region)
        return export_settings

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
            logger.info("output dir already exists. Files in the dir will be overwritten.")
        else:
            output_dir.mkdir()

    def set_run_context(
        self,
        project_name: str,
        study_name: str | None = None,
        restart_target_study_name: str | None = None,
    ) -> None:
        """Store project/study information"""
        self.project_name = project_name
        self.study_name = study_name
        self.restart_target_study_name = restart_target_study_name

    def set_control_provider(self, control_provider: object | None) -> None:
        """Attach an optional external pause/stop provider.

        GUI control still uses multiprocessing Events. Background CLI jobs pass
        a file-backed provider here, allowing the same optimization loop to
        honor `job_pause`, `job_resume`, and `job_stop` without requiring shared
        process memory.
        """
        self.control_provider = control_provider

    def run_optimization(self, summary_dst_dir: str) -> RestartSelection | None:
        """
        Execute optimization

        Args:
            summary_dst_dir (str): destination directory to save summary

        Raises:
            Exception: If an error occurs during processing
        """
        breaked = False
        evaluated_candidates = Population()
        self.current_summary_dir = summary_dst_dir
        self.restart_selection = None
        self.stopped_by_request = False
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
        return self.restart_selection

    def check_optimization_result(self, summary_dst_dir: str) -> RestartSelection | None:
        """
        Check performed optimization result

        Args:
            summary_dst_dir (str): destination directory to load summary
        """
        self.current_summary_dir = summary_dst_dir
        self.restart_selection = None
        try:
            self._setup_visualizer(restart_state="enabled")
            # load result
            base_dir = Path(summary_dst_dir) / BaseConfig.RESULTANT_DIRNAME.value
            filepath = base_dir / self.config.output_control["best_individual"].filename_base
            best_individual_filepath = filepath.with_suffix(".yaml")
            graph_table_enabled = best_individual_filepath.exists()
            graph_table_disabled_reason = None
            if graph_table_enabled:
                individuals = load_individuals(str(best_individual_filepath))
            else:
                logger.warning(
                    "Best individual file is missing at %s. Disable Graph/Table restore and continue check.",
                    best_individual_filepath,
                )
                individuals = []
                graph_table_disabled_reason = f"Best individual file is missing: {best_individual_filepath}"

            # resolve persisted outcome paths against the run summary
            for ind in individuals:
                if ind.outcome_filepath:
                    resolved = resolve_outcome_filepath(ind.outcome_filepath, summary_dir=summary_dst_dir)
                    ind.outcome_filepath = str(resolved) if resolved is not None else None

            response_surface_manifest = self._prepare_response_surface_for_check(summary_dst_dir)

            # Send payload and wait for GUI to close.
            self._send_payload(
                VisualizerPayload(
                    "update",
                    individuals=deepcopy(individuals),
                    graph_table_enabled=graph_table_enabled,
                    graph_table_disabled_reason=graph_table_disabled_reason,
                    response_surface_manifest=response_surface_manifest,
                )
            )
            if self.visualizer_p and self.visualizer_p.is_alive():
                logger.info("To end process, please close GUI (when GUI is displayed).")
                self.visualizer_p.join()
            self._receive_restart_request()

        except Exception as e:
            msg = f"Error during loading result: {e}."
            logger.exception(msg)
            raise RuntimeError(msg) from e

        finally:
            self._finalize_visualizer()
        return self.restart_selection

    def _prepare_response_surface_for_check(self, summary_dst_dir: str) -> dict | None:
        """Initialize the lazy response-surface backend used only during `check`."""
        response_surface_config = getattr(self.config, "response_surface", None)
        if not isinstance(response_surface_config, ResponseSurfaceConfig) or not response_surface_config.enabled:
            return None
        session = build_response_surface_session(summary_dst_dir, response_surface_config)
        if session is None:
            return None
        self.response_surface_session = session
        self._start_response_surface_worker()
        return session.build_manifest()

    def sample(self, num_sample: int, summary_dst_dir: str, chunk_size: int | None = None) -> None:
        """Sample individuals with Latin Hypercube Sampling and persist chunked results.

        The sample command is represented as one study run. Each chunk behaves
        like a pseudo-iteration, so background pause/stop requests are checked
        before starting the next chunk rather than in the middle of an evaluator
        call.
        """
        if num_sample < 1:
            msg = "num_sample must be greater than or equal to 1."
            raise ValueError(msg)
        resolved_chunk_size = num_sample if chunk_size is None else chunk_size
        if resolved_chunk_size < 1:
            msg = "chunk_size must be greater than or equal to 1."
            raise ValueError(msg)

        start = datetime.datetime.now(tz=datetime.UTC)
        variable_dim = self.evaluator.get_variable_dimension()
        sampler = qmc.LatinHypercube(d=variable_dim)
        sample = sampler.random(n=num_sample)
        lower_bounds, upper_bounds = self._resolve_sampling_bounds(variable_dim)
        design_matrix = qmc.scale(sample, lower_bounds, upper_bounds).tolist()
        num_chunks = math.ceil(num_sample / resolved_chunk_size)
        self.current_summary_dir = summary_dst_dir
        self.restart_selection = None
        self.stopped_by_request = False

        self.config.output_interval = 1
        self.config.output_control["best_individual"].enabled = False
        self.config.output_control["candidate"].enabled = True
        self.config.output_control["candidate_plot"].enabled = False

        evaluated_population = Population()
        completed_chunks = 0
        for i_iter, start_idx in enumerate(range(0, num_sample, resolved_chunk_size)):
            # Cooperative background control happens only between chunks so
            # partially evaluated sample groups are not interrupted.
            if self._check_events():
                break
            chunk_matrix = design_matrix[start_idx : start_idx + resolved_chunk_size]
            population = Population.from_design_matrix(chunk_matrix)
            if self.config.enable_parallelization:
                evaluated_population = self.evaluator.evaluate_parallel(population, self.config.num_processes)
            else:
                evaluated_population = self.evaluator.evaluate(population)
            self.time_iterations.append(datetime.datetime.now(tz=datetime.UTC) - start)
            start = datetime.datetime.now(tz=datetime.UTC)
            self._post_iteration(i_iter, evaluated_population)
            completed_chunks += 1

        summary = OptimizationSummary(
            num_iteration=completed_chunks if self.stopped_by_request else num_chunks,
            num_candidates_per_iteration=resolved_chunk_size,
            num_parallel_process=cpu_count()
            if self.config.num_processes is None
            else min(self.config.num_processes, cpu_count()),
            total_computation_time=self.elapsed.total_seconds(),
        )
        self._after_optimization(summary, summary_dst_dir)

    def _resolve_sampling_bounds(self, variable_dim: int) -> tuple[list[float], list[float]]:
        """Resolve per-dimension sampling bounds from optimizer settings."""
        bounds = getattr(self.optimizer, "bounds", None)
        if bounds is None:
            return [-1.0] * variable_dim, [1.0] * variable_dim
        if hasattr(bounds, "tolist"):
            bounds = bounds.tolist()
        if (
            isinstance(bounds, (list, tuple))
            and len(bounds) == BOUNDS_PAIR_LENGTH
            and all(isinstance(value, Real) for value in bounds)
        ):
            return [float(bounds[0])] * variable_dim, [float(bounds[1])] * variable_dim
        if not isinstance(bounds, (list, tuple)):
            msg = "optimizer bounds must be None, a pair of scalars, or a sequence of bound pairs."
            raise TypeError(msg)
        normalized_bounds = [tuple(bound) for bound in bounds]
        if len(normalized_bounds) > variable_dim:
            msg = "optimizer bounds length must be less than or equal to the variable dimension."
            raise ValueError(msg)
        normalized_bounds.extend([(-1.0, 1.0)] * (variable_dim - len(normalized_bounds)))
        lower_bounds = [float(bound[0]) for bound in normalized_bounds]
        upper_bounds = [float(bound[1]) for bound in normalized_bounds]
        return lower_bounds, upper_bounds

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

    def _setup_visualizer(self, restart_state: str = "disabled") -> None:
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
        self.restart_q = ctx.Queue(maxsize=1)
        # GUI sends response-surface requests through this queue while manager keeps the heavy work.
        self.response_surface_request_q = ctx.Queue(maxsize=16)
        self.visualizer_p = ctx.Process(
            target=self.visualizer.run,
            args=(
                mode,
                self.evaluator.get_num_objectives(),
                self.progress_q,
                self.stop_evt,
                self.pause_evt,
                self.restart_q,
                self.response_surface_request_q,
                self.restart_target_study_name,
                restart_state,
            ),
            daemon=True,
        )
        self.visualizer_p.start()

    def _check_events(self) -> bool:
        """Check GUI and background pause/stop signals.

        Returns:
            True when the caller should break its optimization/sample loop.
        """
        if self._stop_requested():
            self.stopped_by_request = True
            return True
        pause_reported = False
        while self._pause_requested():
            if not pause_reported:
                # File-backed controls need an explicit state transition so
                # independent `job_status` commands can see that the worker is paused.
                self._mark_external_paused()
                pause_reported = True
            if self._stop_requested():
                self.stopped_by_request = True
                return True
            time.sleep(0.05)
        if pause_reported:
            self._mark_external_running()
        return False

    def _stop_requested(self) -> bool:
        """Return True when GUI Events or the external provider request stop."""
        if self.stop_evt is not None and self.stop_evt.is_set():
            return True
        if self.control_provider is None:
            return False
        stop_requested = getattr(self.control_provider, "stop_requested", None)
        if not callable(stop_requested):
            return False
        return bool(stop_requested())

    def _pause_requested(self) -> bool:
        """Return True when GUI Events or the external provider request pause."""
        if self.pause_evt is not None and self.pause_evt.is_set():
            return True
        if self.control_provider is None:
            return False
        pause_requested = getattr(self.control_provider, "pause_requested", None)
        if not callable(pause_requested):
            return False
        return bool(pause_requested())

    def _mark_external_paused(self) -> None:
        """Notify an external provider that the manager has entered pause wait."""
        if self.control_provider is None:
            return
        mark_paused = getattr(self.control_provider, "mark_paused", None)
        if callable(mark_paused):
            mark_paused()

    def _mark_external_running(self) -> None:
        """Notify an external provider that the manager has resumed from pause."""
        if self.control_provider is None:
            return
        mark_running = getattr(self.control_provider, "mark_running", None)
        if callable(mark_running):
            mark_running()

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
        logger.info("Iteration %d finished", i_iter + 1)

    def _after_optimization(self, summary: OptimizationSummary, summary_dst_dir: str) -> None:
        """After optimization pocess"""
        # Save summary
        if Path(summary_dst_dir).exists():
            logger.warning("Optimization summary of this project already exists. Overwriting...")
            shutil.rmtree(summary_dst_dir)
        Path(summary_dst_dir).mkdir(parents=True)
        filepath = Path(summary_dst_dir) / BaseConfig.SUMMARY_FILENAME.value
        with filepath.open(mode="w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(summary), f)
        # copy output dir
        self.output_handler.save_latest_summary_dir(str(Path(summary_dst_dir)))
        # Wait for GUI to close.
        if self.visualizer_p and self.visualizer_p.is_alive():
            if self.progress_q:
                self._send_payload(VisualizerPayload(type="restart_ready"))
            logger.info("Optimization finished. To end process, please close GUI (when GUI is displayed).")
            self.visualizer_p.join()
            self._receive_restart_request()

    def _start_response_surface_worker(self) -> None:
        """Start the background thread that translates GUI requests into response-surface payloads."""
        if self.response_surface_session is None or self.response_surface_request_q is None:
            return
        if self._response_surface_worker_thread and self._response_surface_worker_thread.is_alive():
            return
        self._response_surface_worker_stop.clear()
        self._response_surface_worker_thread = threading.Thread(
            target=self._response_surface_worker_loop,
            name="response-surface-worker",
            daemon=True,
        )
        self._response_surface_worker_thread.start()

    def _response_surface_worker_loop(self) -> None:
        """Serve GUI response-surface requests until the visualizer or worker is stopped."""
        while not self._response_surface_worker_stop.is_set():
            if self.visualizer_p is not None and not self.visualizer_p.is_alive():
                return
            if self.response_surface_request_q is None or self.response_surface_session is None:
                return
            try:
                request = self.response_surface_request_q.get(timeout=0.1)
            except queue.Empty:
                continue
            except (EOFError, OSError, ValueError):
                return
            if not isinstance(request, ResponseSurfaceRequest):
                continue
            try:
                # The session owns both the RF model cache and the contour-grid cache.
                result = self.response_surface_session.create_response(request)
            except (ValueError, RuntimeError) as e:
                logger.warning("Response surface generation failed: %s", e)
                result = {"request_id": request.request_id, "error": str(e)}
            self._send_payload(VisualizerPayload(type="response_surface_ready", response_surface_result=result))

    def _stop_response_surface_worker(self) -> None:
        """Stop the response-surface thread and drop session-scoped caches."""
        self._response_surface_worker_stop.set()
        if self._response_surface_worker_thread and self._response_surface_worker_thread.is_alive():
            self._response_surface_worker_thread.join(timeout=1.0)
        self._response_surface_worker_thread = None
        self.response_surface_session = None

    def _finalize_visualizer(self) -> None:
        """Finalize: clean up visualizer process and resources."""
        logger.info("Gracefully closing... (Press Ctrl + C to terminate)")
        try:
            self._stop_response_surface_worker()
            if self.visualizer_p and self.visualizer_p.is_alive():
                if self.progress_q:
                    self._send_payload(VisualizerPayload(type="done"))
                self.visualizer_p.join()
            if self.progress_q:
                self.progress_q.close()
                self.progress_q.join_thread()
            if self.restart_q:
                self.restart_q.close()
                self.restart_q.join_thread()
            if self.response_surface_request_q:
                self.response_surface_request_q.close()
                self.response_surface_request_q.join_thread()
                self.response_surface_request_q = None
        except KeyboardInterrupt:
            self.visualizer_p.kill()

    def _receive_restart_request(self) -> RestartSelection | None:
        """Receive a restart selection from the GUI if available."""
        if self.restart_q is None or self.restart_selection is not None:
            return self.restart_selection
        try:
            self.restart_selection = self.restart_q.get_nowait()
        except queue.Empty:
            return None
        return self.restart_selection

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
        except Exception as e:  # noqa: BLE001 ; continue optimization run anyway.
            logger.warning("Output progress failed. Continue optimization run.")
            logger.warning("Error message: %s", str(e))
