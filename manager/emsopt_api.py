"""
emsopt_api.py
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

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import utils.emsopt_cli_helpers as cli_helpers
from emsopt_engine.setup_project import setup_project_files
from manager.background_job import BackgroundJobManager
from manager.optimization_manager import OptimizationManager
from manager.restart_selection import RestartSelection
from manager.run_analysis import write_run_info


@dataclass
class ManagerHandle:
    """A prepared OptimizationManager and its resolved project/study context."""

    project_name: str
    study_name: str
    manager: OptimizationManager
    restart_target_study_name: str | None = None


@dataclass
class OptimizationCommandResult:
    """Result returned by the public API after a manager-backed command."""

    project_name: str
    study_name: str
    status: str
    manager: OptimizationManager
    run_id: str | None = None
    run_summary_dir: Path | None = None
    restart_selection: RestartSelection | None = None
    failure_reason: str | None = None


class EMSOptimizerClient:
    """High-level Python API for EMSOptimizer project, study, and run operations."""

    def __init__(
        self,
        *,
        project_root: str | Path | None = None,
    ) -> None:
        self.project_root = None if project_root is None else Path(project_root)

    def create_manager(
        self,
        project_name: str,
        study_name: str | None = None,
        *,
        restart_target_study_name: str | None = None,
    ) -> ManagerHandle:
        """Create an OptimizationManager and expose it for external scripts."""
        resolved_study_name = cli_helpers.resolve_effective_study_name(
            project_name,
            study_name,
            self.project_root,
        )
        resolved_restart_target = restart_target_study_name
        if resolved_restart_target is None:
            resolved_restart_target = cli_helpers.resolve_restart_target_study_name(
                project_name,
                resolved_study_name,
                self.project_root,
            )
        manager = self._build_manager(project_name, resolved_study_name, resolved_restart_target)
        return ManagerHandle(
            project_name=project_name,
            study_name=resolved_study_name,
            manager=manager,
            restart_target_study_name=resolved_restart_target,
        )

    def run(
        self,
        project_name: str,
        study_name: str | None = None,
        *,
        control_provider: object | None = None,
        disable_progress_gui: bool = False,
        run_started_callback: Callable[[str, Path], None] | None = None,
    ) -> OptimizationCommandResult:
        """Run optimization for a study, including restart handling and finalization.

        Args:
            project_name: Target project identifier.
            study_name: Target study name. `None` resolves to the default study.
            control_provider: Optional background-job control adapter. When set,
                pause/stop requests are checked inside `OptimizationManager`.
            disable_progress_gui: Disable the interactive progress GUI. Background
                jobs use this to avoid opening UI from the worker process.
            run_started_callback: Optional callback invoked immediately after a
                run id and run summary directory are allocated.
        """
        current_study_name = cli_helpers.resolve_effective_study_name(
            project_name,
            study_name,
            self.project_root,
        )
        latest_result: OptimizationCommandResult | None = None
        while True:
            restart_target_study_name = cli_helpers.resolve_restart_target_study_name(
                project_name,
                current_study_name,
                self.project_root,
            )
            manager = self._build_manager(project_name, current_study_name, restart_target_study_name)
            if disable_progress_gui:
                manager.config.enable_progress_gui = False
            manager.set_control_provider(control_provider)
            run_id, run_summary_dir, started_at = self._start_run(project_name, current_study_name)
            if run_started_callback is not None:
                # Background jobs use this callback to expose the live run id via job_status.
                run_started_callback(run_id, run_summary_dir)
            try:
                restart_selection = manager.run_optimization(str(run_summary_dir))
                self._finalize(project_name, current_study_name, run_summary_dir, run_id)
                if manager.stopped_by_request is True:
                    self._write_run_stopped(run_summary_dir, run_id, current_study_name, started_at)
                else:
                    self._write_run_completed(run_summary_dir, run_id, current_study_name, started_at)
            except Exception as e:
                self._write_run_failed(run_summary_dir, run_id, current_study_name, started_at, str(e))
                raise
            status = "stopped" if manager.stopped_by_request is True else "completed"
            latest_result = OptimizationCommandResult(
                project_name=project_name,
                study_name=current_study_name,
                run_id=run_id,
                run_summary_dir=run_summary_dir,
                status=status,
                manager=manager,
                restart_selection=restart_selection,
            )
            if not isinstance(restart_selection, RestartSelection):
                return latest_result
            current_study_name = cli_helpers.apply_restart_selection(
                project_name,
                current_study_name,
                restart_selection,
                self.project_root,
            )

    def check(
        self,
        project_name: str,
        study_name: str | None = None,
        run_id: str | None = None,
    ) -> OptimizationCommandResult:
        """Open/check an optimization result for a study."""
        current_study_name = cli_helpers.resolve_effective_study_name(
            project_name,
            study_name,
            self.project_root,
        )
        try:
            if run_id is None:
                summary_dir = cli_helpers.get_latest_run_summary_dir(
                    project_name,
                    current_study_name,
                    self.project_root,
                )
            else:
                summary_dir = cli_helpers.get_existing_run_summary_dir(
                    project_name,
                    current_study_name,
                    run_id,
                    self.project_root,
                )
        except (FileNotFoundError, ValueError) as e:
            msg = f"Error during loading result: {e}."
            raise RuntimeError(msg) from e

        restart_target_study_name = cli_helpers.resolve_restart_target_study_name(
            project_name,
            current_study_name,
            self.project_root,
        )
        manager = self._build_manager(project_name, current_study_name, restart_target_study_name)
        restart_selection = manager.check_optimization_result(str(summary_dir))
        result = OptimizationCommandResult(
            project_name=project_name,
            study_name=current_study_name,
            run_id=self._infer_run_id(summary_dir),
            run_summary_dir=summary_dir,
            status="completed",
            manager=manager,
            restart_selection=restart_selection,
        )
        if isinstance(restart_selection, RestartSelection):
            target_study_name = cli_helpers.apply_restart_selection(
                project_name,
                current_study_name,
                restart_selection,
                self.project_root,
            )
            self.run(project_name, target_study_name)
        return result

    def sample(
        self,
        project_name: str,
        num_sample: int,
        study_name: str | None = None,
        *,
        chunk_size: int | None = None,
        control_provider: object | None = None,
        run_started_callback: Callable[[str, Path], None] | None = None,
    ) -> OptimizationCommandResult:
        """Evaluate an LHS sample batch and persist it as a study run.

        `sample` follows the same run storage contract as `run`: it allocates a
        run id, writes `run_info.yaml`, finalizes study-level artifacts, and can
        be controlled by a background job provider at sample-chunk boundaries.
        """
        current_study_name = cli_helpers.resolve_effective_study_name(
            project_name,
            study_name,
            self.project_root,
        )
        manager = self._build_manager(project_name, current_study_name)
        manager.set_control_provider(control_provider)
        run_id, run_summary_dir, started_at = self._start_run(project_name, current_study_name)
        if run_started_callback is not None:
            # Keep job_info.yaml aligned with the run directory created for this sample batch.
            run_started_callback(run_id, run_summary_dir)
        try:
            manager.sample(num_sample, str(run_summary_dir), chunk_size=chunk_size)
            self._finalize(project_name, current_study_name, run_summary_dir, run_id)
            if manager.stopped_by_request is True:
                self._write_run_stopped(run_summary_dir, run_id, current_study_name, started_at)
            else:
                self._write_run_completed(run_summary_dir, run_id, current_study_name, started_at)
        except Exception as e:
            self._write_run_failed(run_summary_dir, run_id, current_study_name, started_at, str(e))
            raise
        status = "stopped" if manager.stopped_by_request is True else "completed"
        return OptimizationCommandResult(
            project_name=project_name,
            study_name=current_study_name,
            run_id=run_id,
            run_summary_dir=run_summary_dir,
            status=status,
            manager=manager,
        )

    def batch_run(
        self,
        project_name: str,
        num_runs: int,
        study_name: str | None = None,
        *,
        stop_on_error: bool = False,
        control_provider: object | None = None,
        run_started_callback: Callable[[str, Path], None] | None = None,
    ) -> list[OptimizationCommandResult]:
        """Run the same study multiple times as independent runs.

        When a control provider is attached, the stop request is checked before
        starting each new run and again inside each `OptimizationManager` loop.
        This keeps `batch_run --background` from launching further runs after a
        cooperative stop has been requested.
        """
        current_study_name = cli_helpers.resolve_effective_study_name(
            project_name,
            study_name,
            self.project_root,
        )
        if num_runs < 1:
            msg = "num_runs must be greater than or equal to 1."
            raise ValueError(msg)

        results: list[OptimizationCommandResult] = []
        for _ in range(num_runs):
            if self._control_stop_requested(control_provider):
                break
            manager = self._build_manager(project_name, current_study_name)
            manager.config.enable_progress_gui = False
            manager.set_control_provider(control_provider)
            run_id, run_summary_dir, started_at = self._start_run(project_name, current_study_name)
            if run_started_callback is not None:
                run_started_callback(run_id, run_summary_dir)
            try:
                manager.run_optimization(str(run_summary_dir))
                self._finalize(project_name, current_study_name, run_summary_dir, run_id)
                if manager.stopped_by_request is True:
                    self._write_run_stopped(run_summary_dir, run_id, current_study_name, started_at)
                else:
                    self._write_run_completed(run_summary_dir, run_id, current_study_name, started_at)
            except Exception as e:
                failure_reason = str(e)
                self._write_run_failed(run_summary_dir, run_id, current_study_name, started_at, failure_reason)
                results.append(
                    OptimizationCommandResult(
                        project_name=project_name,
                        study_name=current_study_name,
                        run_id=run_id,
                        run_summary_dir=run_summary_dir,
                        status="failed",
                        manager=manager,
                        failure_reason=failure_reason,
                    )
                )
                if stop_on_error:
                    raise
            else:
                status = "stopped" if manager.stopped_by_request is True else "completed"
                results.append(
                    OptimizationCommandResult(
                        project_name=project_name,
                        study_name=current_study_name,
                        run_id=run_id,
                        run_summary_dir=run_summary_dir,
                        status=status,
                        manager=manager,
                    )
                )
                if manager.stopped_by_request is True:
                    break
        return results

    def start_background_run(self, project_name: str, study_name: str | None = None) -> dict:
        """Start `run` in a detached worker process and return job metadata."""
        return (
            BackgroundJobManager(self.project_root)
            .start(
                command="run",
                project_name=project_name,
                study_name=study_name,
            )
            .to_dict()
        )

    def start_background_batch_run(
        self,
        project_name: str,
        num_runs: int,
        study_name: str | None = None,
        *,
        stop_on_error: bool = False,
    ) -> dict:
        """Start `batch_run` in a detached worker process and return job metadata."""
        if num_runs < 1:
            msg = "num_runs must be greater than or equal to 1."
            raise ValueError(msg)
        return (
            BackgroundJobManager(self.project_root)
            .start(
                command="batch_run",
                project_name=project_name,
                study_name=study_name,
                num_runs=num_runs,
                stop_on_error=stop_on_error,
            )
            .to_dict()
        )

    def start_background_sample(
        self,
        project_name: str,
        num_sample: int,
        study_name: str | None = None,
        *,
        chunk_size: int | None = None,
    ) -> dict:
        """Start `sample` in a detached worker process and return job metadata."""
        if num_sample < 1:
            msg = "num_sample must be greater than or equal to 1."
            raise ValueError(msg)
        if chunk_size is not None and chunk_size < 1:
            msg = "chunk_size must be greater than or equal to 1."
            raise ValueError(msg)
        return (
            BackgroundJobManager(self.project_root)
            .start(
                command="sample",
                project_name=project_name,
                study_name=study_name,
                num_sample=num_sample,
                chunk_size=chunk_size,
            )
            .to_dict()
        )

    def background_job_status(self, project_name: str, study_name: str | None, job_id: str) -> dict:
        """Return persisted status for one background job."""
        return BackgroundJobManager(self.project_root).status(
            project_name=project_name,
            study_name=study_name,
            job_id=job_id,
        )

    def list_background_jobs(self, project_name: str, study_name: str | None = None) -> list[dict]:
        """Return persisted status records for all jobs in a project/study."""
        return BackgroundJobManager(self.project_root).list_jobs(project_name=project_name, study_name=study_name)

    def pause_background_job(self, project_name: str, study_name: str | None, job_id: str) -> dict:
        """Request pause for a background job and return its updated status."""
        return BackgroundJobManager(self.project_root).pause(
            project_name=project_name,
            study_name=study_name,
            job_id=job_id,
        )

    def resume_background_job(self, project_name: str, study_name: str | None, job_id: str) -> dict:
        """Request resume for a background job and return its updated status."""
        return BackgroundJobManager(self.project_root).resume(
            project_name=project_name,
            study_name=study_name,
            job_id=job_id,
        )

    def stop_background_job(self, project_name: str, study_name: str | None, job_id: str) -> dict:
        """Request cooperative stop for a background job and return its updated status."""
        return BackgroundJobManager(self.project_root).stop(
            project_name=project_name,
            study_name=study_name,
            job_id=job_id,
        )

    def _build_manager(
        self,
        project_name: str,
        study_name: str,
        restart_target_study_name: str | None = None,
    ) -> OptimizationManager:
        if self.project_root is None:
            injector = setup_project_files(project_name, study_name)
        else:
            injector = setup_project_files(project_name, study_name, project_root=self.project_root)
        manager = injector.get(OptimizationManager)
        manager.set_run_context(project_name, study_name, restart_target_study_name)
        return manager

    def _start_run(self, project_name: str, study_name: str) -> tuple[str, Path, datetime]:
        run_id = cli_helpers.allocate_run_id(project_name, study_name, self.project_root)
        run_summary_dir = cli_helpers.get_run_summary_dir(
            project_name,
            study_name,
            run_id,
            self.project_root,
        )
        started_at = datetime.now(UTC)
        write_run_info(
            run_summary_dir,
            run_id=run_id,
            study_name=study_name,
            status="running",
            started_at=started_at,
        )
        return run_id, run_summary_dir, started_at

    def _finalize(self, project_name: str, study_name: str, run_summary_dir: Path, run_id: str) -> None:
        cli_helpers.finalize_study_run(project_name, study_name, run_summary_dir, run_id, self.project_root)

    @staticmethod
    def _write_run_completed(run_summary_dir: Path, run_id: str, study_name: str, started_at: datetime) -> None:
        write_run_info(
            run_summary_dir,
            run_id=run_id,
            study_name=study_name,
            status="completed",
            started_at=started_at,
            finished_at=datetime.now(UTC),
        )

    @staticmethod
    def _write_run_failed(
        run_summary_dir: Path,
        run_id: str,
        study_name: str,
        started_at: datetime,
        failure_reason: str,
    ) -> None:
        write_run_info(
            run_summary_dir,
            run_id=run_id,
            study_name=study_name,
            status="failed",
            started_at=started_at,
            finished_at=datetime.now(UTC),
            failure_reason=failure_reason,
        )

    @staticmethod
    def _write_run_stopped(run_summary_dir: Path, run_id: str, study_name: str, started_at: datetime) -> None:
        """Persist run metadata for a run ended by a cooperative stop request."""
        write_run_info(
            run_summary_dir,
            run_id=run_id,
            study_name=study_name,
            status="stopped",
            started_at=started_at,
            finished_at=datetime.now(UTC),
        )

    @staticmethod
    def _infer_run_id(summary_dir: Path) -> str | None:
        if summary_dir.name.startswith("run_"):
            return summary_dir.name
        return None

    @staticmethod
    def _control_stop_requested(control_provider: object | None) -> bool:
        """Read a stop request from an optional background control provider."""
        if control_provider is None:
            return False
        stop_requested = getattr(control_provider, "stop_requested", None)
        if not callable(stop_requested):
            return False
        return bool(stop_requested())
