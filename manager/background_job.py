"""
background_job.py
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

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import yaml

import utils.emsopt_cli_helpers as cli_helpers
from emsopt_engine.configs.base import BaseConfig

JOB_INFO_FILENAME = "job_info.yaml"
JOB_CONTROL_FILENAME = "control.yaml"
JOBS_DIRNAME = "jobs"
JOB_STATUS_ACTIVE = {"queued", "running", "pausing", "paused", "stopping"}


@dataclass
class BackgroundJobStartResult:
    """Launch metadata returned to CLI/API callers after a worker process starts.

    Attributes:
        job_id: Stable identifier used by `job_status`, `job_pause`,
            `job_resume`, and `job_stop`.
        pid: Operating-system process id for the spawned worker process.
        job_dir: Directory containing `job_info.yaml`, `control.yaml`, and logs.
        status: Initial user-facing job status, normally `running`.
    """

    job_id: str
    pid: int
    job_dir: Path
    status: str

    def to_dict(self) -> dict[str, object]:
        """Convert the dataclass to the JSON-serializable CLI/API payload."""
        return {
            "job_id": self.job_id,
            "pid": self.pid,
            "job_dir": str(self.job_dir),
            "status": self.status,
        }


class FileJobControl:
    """Control adapter consumed by `OptimizationManager` during a background job.

    The adapter intentionally exposes the same small protocol that
    `OptimizationManager` needs: `stop_requested`, `pause_requested`,
    `mark_paused`, and `mark_running`. It translates those calls into reads and
    writes of `control.yaml` and `job_info.yaml`, so separate CLI invocations can
    control a process without sharing Python objects or multiprocessing Events.
    """

    def __init__(self, job_dir: str | Path) -> None:
        """Initialize the adapter for one existing background job directory."""
        self.job_dir = Path(job_dir)
        self.control_path = self.job_dir / JOB_CONTROL_FILENAME
        self.info_path = self.job_dir / JOB_INFO_FILENAME
        self._pause_reported = False
        self._stopping_reported = False

    def stop_requested(self) -> bool:
        """Return True when `job_stop` has requested cooperative termination."""
        payload = self._read_control()
        requested = bool(payload.get("stop_requested", False))
        if requested and not self._stopping_reported:
            self.update_job_info(status="stopping")
            self._stopping_reported = True
        return requested

    def pause_requested(self) -> bool:
        """Return True while `job_pause` is requesting the manager to wait."""
        return bool(self._read_control().get("pause_requested", False))

    def mark_paused(self) -> None:
        """Record that the running manager has reached the paused state."""
        if self._pause_reported:
            return
        self.update_job_info(status="paused")
        self._pause_reported = True

    def mark_running(self) -> None:
        """Record that the manager has left a pause and resumed execution."""
        if not self._pause_reported:
            return
        self.update_job_info(status="running")
        self._pause_reported = False

    def update_job_info(self, **updates: object) -> dict[str, object]:
        """Merge fields into `job_info.yaml` and refresh its `updated_at` value."""
        payload = read_yaml_dict(self.info_path)
        payload.update(updates)
        payload["updated_at"] = _now_iso()
        write_yaml_dict(self.info_path, payload)
        return payload

    def _read_control(self) -> dict[str, object]:
        """Read the current control request payload, returning an empty dict if absent."""
        return read_yaml_dict(self.control_path)


class BackgroundJobManager:
    """Create, inspect, and control background jobs for one project root.

    The manager is used in two different processes:
    - foreground CLI/API calls use `start`, `status`, `list_jobs`, `pause`,
      `resume`, and `stop`;
    - the spawned worker process uses `run_job`.

    Keeping these operations in one class makes the on-disk job contract
    explicit and avoids duplicating path resolution around the CLI layer.
    """

    def __init__(self, project_root: str | Path | None = None) -> None:
        """Store the optional project root used for all project/study lookups."""
        self.project_root = None if project_root is None else Path(project_root)

    def start(
        self,
        *,
        command: str,
        project_name: str,
        study_name: str | None = None,
        num_runs: int | None = None,
        num_sample: int | None = None,
        chunk_size: int | None = None,
        stop_on_error: bool = False,
    ) -> BackgroundJobStartResult:
        """Create job files and spawn the worker process.

        Args:
            command: Background command to execute. Supported values are
                `run`, `batch_run`, and `sample`.
            project_name: Target EMSOptimizer project identifier.
            study_name: Target study name. `None` follows the usual default
                study resolution rules.
            num_runs: Number of independent runs for `batch_run`.
            num_sample: Number of LHS samples for `sample`.
            chunk_size: Number of samples per pseudo-iteration for `sample`.
            stop_on_error: Whether `batch_run` stops after the first failed run.

        Returns:
            Launch metadata including job id, pid, job directory, and status.
        """
        resolved_study_name = cli_helpers.resolve_effective_study_name(project_name, study_name, self.project_root)
        job_id = self._allocate_job_id(project_name, resolved_study_name)
        job_dir = self._job_dir(project_name, resolved_study_name, job_id)
        job_dir.mkdir(parents=True, exist_ok=False)
        now = _now_iso()
        write_yaml_dict(
            job_dir / JOB_INFO_FILENAME,
            {
                "job_id": job_id,
                "project_name": project_name,
                "study_name": resolved_study_name,
                "project_root": None if self.project_root is None else str(self.project_root),
                "command": command,
                "pid": None,
                "status": "queued",
                "num_runs": num_runs,
                "num_sample": num_sample,
                "chunk_size": chunk_size,
                "stop_on_error": bool(stop_on_error),
                "run_ids": [],
                "current_run_id": None,
                "started_at": now,
                "finished_at": None,
                "failure_reason": None,
                "updated_at": now,
            },
        )
        write_yaml_dict(
            job_dir / JOB_CONTROL_FILENAME,
            {
                "pause_requested": False,
                "stop_requested": False,
                "updated_at": now,
            },
        )

        # The parent records pid/status only after Popen succeeds, so an
        # incomplete job directory is easy to distinguish from a live job.
        process = self._spawn_worker(job_dir)
        payload = read_yaml_dict(job_dir / JOB_INFO_FILENAME)
        payload.update({"pid": process.pid, "status": "running", "updated_at": _now_iso()})
        write_yaml_dict(job_dir / JOB_INFO_FILENAME, payload)
        return BackgroundJobStartResult(job_id=job_id, pid=process.pid, job_dir=job_dir, status="running")

    def status(self, *, project_name: str, study_name: str | None, job_id: str) -> dict[str, object]:
        """Return one job's latest persisted status payload."""
        resolved_study_name = cli_helpers.resolve_effective_study_name(project_name, study_name, self.project_root)
        job_dir = self._job_dir(project_name, resolved_study_name, job_id)
        if not job_dir.exists():
            msg = f"Background job '{job_id}' does not exist for study '{resolved_study_name}'."
            raise FileNotFoundError(msg)
        return self._read_status(job_dir)

    def list_jobs(self, *, project_name: str, study_name: str | None) -> list[dict[str, object]]:
        """Return all persisted background jobs for the resolved project/study."""
        resolved_study_name = cli_helpers.resolve_effective_study_name(project_name, study_name, self.project_root)
        jobs_dir = self._jobs_dir(project_name, resolved_study_name)
        if not jobs_dir.exists():
            return []
        return [
            self._read_status(job_dir) for job_dir in sorted(child for child in jobs_dir.iterdir() if child.is_dir())
        ]

    def pause(self, *, project_name: str, study_name: str | None, job_id: str) -> dict[str, object]:
        """Request a running job to pause at its next safe control point."""
        return self._request_control(project_name, study_name, job_id, pause_requested=True, status="pausing")

    def resume(self, *, project_name: str, study_name: str | None, job_id: str) -> dict[str, object]:
        """Request a paused job to continue from its control wait loop."""
        return self._request_control(project_name, study_name, job_id, pause_requested=False, status="running")

    def stop(self, *, project_name: str, study_name: str | None, job_id: str) -> dict[str, object]:
        """Request a job to stop cooperatively at its next safe control point."""
        return self._request_control(project_name, study_name, job_id, stop_requested=True, status="stopping")

    def run_job(self, job_dir: str | Path) -> None:
        """Execute the command described by `job_info.yaml` inside a child process.

        This method is called only from the internal `_run_background_job` CLI
        command. It reconstructs an `EMSOptimizerClient`, attaches
        `FileJobControl` to the underlying manager, and keeps `job_info.yaml`
        synchronized with the current run id while the operation is active.
        """
        job_dir = Path(job_dir)
        info_path = job_dir / JOB_INFO_FILENAME
        info = read_yaml_dict(info_path)
        control = FileJobControl(job_dir)
        control.update_job_info(pid=os.getpid(), status="running")
        try:
            from manager.emsopt_api import EMSOptimizerClient

            client = EMSOptimizerClient(project_root=info.get("project_root"))
            command = str(info["command"])
            project_name = str(info["project_name"])
            study_name = str(info["study_name"])

            def _on_run_started(run_id: str, run_summary_dir: Path) -> None:
                """Record the current run so `job_status` can show live progress."""
                current = read_yaml_dict(info_path)
                run_ids = list(current.get("run_ids") or [])
                if run_id not in run_ids:
                    run_ids.append(run_id)
                current.update(
                    {
                        "run_ids": run_ids,
                        "current_run_id": run_id,
                        "current_run_summary_dir": str(run_summary_dir),
                        "updated_at": _now_iso(),
                    }
                )
                write_yaml_dict(info_path, current)

            self._run_supported_command(
                client,
                control,
                command,
                project_name,
                study_name,
                int(info.get("num_runs") or 0),
                int(info.get("num_sample") or 0),
                None if info.get("chunk_size") is None else int(info["chunk_size"]),
                stop_on_error=bool(info.get("stop_on_error", False)),
                run_started_callback=_on_run_started,
            )
        except Exception as exc:
            control.update_job_info(status="failed", finished_at=_now_iso(), failure_reason=str(exc))
            raise
        else:
            final_status = "stopped" if control.stop_requested() else "completed"
            control.update_job_info(status=final_status, current_run_id=None, finished_at=_now_iso())

    def _request_control(
        self,
        project_name: str,
        study_name: str | None,
        job_id: str,
        *,
        status: str,
        **updates: object,
    ) -> dict[str, object]:
        """Update `control.yaml`, optimistically update job state, and return status."""
        resolved_study_name = cli_helpers.resolve_effective_study_name(project_name, study_name, self.project_root)
        job_dir = self._job_dir(project_name, resolved_study_name, job_id)
        if not job_dir.exists():
            msg = f"Background job '{job_id}' does not exist for study '{resolved_study_name}'."
            raise FileNotFoundError(msg)
        control_path = job_dir / JOB_CONTROL_FILENAME
        control = read_yaml_dict(control_path)
        control.update(updates)
        control["updated_at"] = _now_iso()
        write_yaml_dict(control_path, control)
        info = read_yaml_dict(job_dir / JOB_INFO_FILENAME)
        if str(info.get("status")) in JOB_STATUS_ACTIVE:
            info["status"] = status
            info["updated_at"] = _now_iso()
            write_yaml_dict(job_dir / JOB_INFO_FILENAME, info)
        return self._read_status(job_dir)

    def _read_status(self, job_dir: Path) -> dict[str, object]:
        """Read `job_info.yaml` and mark active jobs as `lost` if their pid is gone."""
        info = read_yaml_dict(job_dir / JOB_INFO_FILENAME)
        info["job_dir"] = str(job_dir)
        status = str(info.get("status", "unknown"))
        pid = info.get("pid")
        if status in JOB_STATUS_ACTIVE and isinstance(pid, int) and not _pid_exists(pid):
            info["status"] = "lost"
            info["updated_at"] = _now_iso()
            write_yaml_dict(job_dir / JOB_INFO_FILENAME, {k: v for k, v in info.items() if k != "job_dir"})
        return info

    def _spawn_worker(self, job_dir: Path) -> subprocess.Popen:
        """Launch the detached Python worker and redirect its output to job logs."""
        stdout_path = job_dir / "stdout.log"
        stderr_path = job_dir / "stderr.log"
        root_script = Path(__file__).resolve().parents[1] / "emsopt.py"
        args = [sys.executable, str(root_script), "_run_background_job", str(job_dir)]
        stdout = stdout_path.open("ab")
        stderr = stderr_path.open("ab")
        kwargs: dict[str, object] = {"stdout": stdout, "stderr": stderr, "stdin": subprocess.DEVNULL}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        try:
            return subprocess.Popen(args, **kwargs)  # noqa: S603
        finally:
            stdout.close()
            stderr.close()

    def _allocate_job_id(self, project_name: str, study_name: str) -> str:
        """Allocate a timestamped job id that does not collide in the jobs directory."""
        jobs_dir = self._jobs_dir(project_name, study_name)
        while True:
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            job_id = f"job_{timestamp}_{uuid4().hex[:8]}"
            if not (jobs_dir / job_id).exists():
                return job_id
            time.sleep(0.001)

    def _jobs_dir(self, project_name: str, study_name: str) -> Path:
        """Return the directory that stores all jobs for one project/study."""
        project_dir = cli_helpers.get_project_dir(project_name, self.project_root)
        return (
            project_dir
            / BaseConfig.SUMMARY_DIRNAME.value
            / BaseConfig.OPTIMIZATION_STUDIES_DIRNAME.value
            / study_name
            / JOBS_DIRNAME
        )

    def _job_dir(self, project_name: str, study_name: str, job_id: str) -> Path:
        """Return the directory for one concrete background job id."""
        return self._jobs_dir(project_name, study_name) / job_id

    @staticmethod
    def _run_supported_command(
        client: object,
        control: FileJobControl,
        command: str,
        project_name: str,
        study_name: str,
        num_runs: int,
        num_sample: int,
        chunk_size: int | None,
        *,
        stop_on_error: bool,
        run_started_callback: object,
    ) -> None:
        """Dispatch a supported background command to `EMSOptimizerClient`.

        The foreground process stores command-specific arguments in
        `job_info.yaml`; this helper maps those generic fields back to the
        corresponding client method. Unsupported command names are treated as
        programming errors in the internal worker command.
        """
        if command == "run":
            client.run(
                project_name,
                study_name,
                control_provider=control,
                disable_progress_gui=True,
                run_started_callback=run_started_callback,
            )
            return
        if command == "sample":
            client.sample(
                project_name,
                num_sample,
                study_name,
                chunk_size=chunk_size,
                control_provider=control,
                run_started_callback=run_started_callback,
            )
            return
        if command == "batch_run":
            client.batch_run(
                project_name,
                num_runs,
                study_name,
                stop_on_error=stop_on_error,
                control_provider=control,
                run_started_callback=run_started_callback,
            )
            return
        msg = f"Unsupported background command: {command}"
        raise ValueError(msg)


def read_yaml_dict(path: str | Path) -> dict[str, object]:
    """Read a YAML mapping from disk, returning `{}` for missing or non-mapping files."""
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    return payload if isinstance(payload, dict) else {}


def write_yaml_dict(path: str | Path, payload: dict[str, object]) -> None:
    """Atomically write a YAML mapping by replacing the target with a temp file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    tmp_path.replace(path)


def run_background_job_cli(job_dir: str) -> None:
    """CLI-facing wrapper for the internal `_run_background_job` subcommand."""
    BackgroundJobManager().run_job(job_dir)


def _now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 form for persisted metadata."""
    return datetime.now(UTC).isoformat()


def _pid_exists(pid: int) -> bool:
    """Return whether an operating-system process id appears to still exist."""
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True
