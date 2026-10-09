"""
emsopt_cli_helpers.py
The MIT License (MIT)
Copyright (c) 2026 Science Solutions International Laboratory, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import json
import re
import shutil
import sys
from datetime import UTC, datetime
from logging import getLogger
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from emsopt_engine.configs.base import BaseConfig
from emsopt_engine.configs.machine import MachineConfig
from emsopt_engine.configs.optimization import OptimizationConfig
from emsopt_engine.configs.optimization_problem import OptimizationProblemConfig
from emsopt_engine.project_locator import ProjectLocator, resolve_project_root
from emsopt_engine.setup_project import get_study_dir, setup_project_files
from manager.optimization_output_handler import append_cross_study_table, output_surrogate_result_artifacts
from manager.restart_selection import RestartSelection
from manager.run_analysis import analyze_study_runs, build_multi_run_visualizer_data

DEFAULT_STUDY_NAME = "default"
logger = getLogger(__name__)


def get_project_root(project_root: str | Path | None = None) -> Path:
    """Return the root directory that contains EMSOptimizer projects."""
    return resolve_project_root(project_root)


def get_project_dir(
    project_name: str,
    project_root: str | Path | None = None,
) -> Path:
    """Return project_root/project_name after validating project_name."""
    return ProjectLocator.from_values(project_name, project_root).project_dir


def call_project_setup(
    project_name: str,
    study_name: str,
    project_root: str | Path | None = None,
) -> object:
    """Call setup_project_files without a project_root kwarg when the default root is used."""
    if project_root is None:
        return setup_project_files(project_name, study_name)
    return setup_project_files(project_name, study_name, project_root=project_root)


def write_json_result(result: dict) -> None:
    """Write a machine-readable JSON result to stdout."""
    sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2))
    sys.stdout.write("\n")


def emit_command_result(command: str, result: dict) -> None:
    """Write a CLI command result when the function was invoked through main()."""
    if command is None:
        return
    write_json_result({"command": command, "success": True, **result})


def emit_command_failure(command: str, exc: Exception) -> None:
    """Write a CLI command failure result to stdout."""
    write_json_result(
        {
            "command": command,
            "success": False,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
    )


def confirm_irreversible_action(prompt: str, *, skip_confirmation: bool = False) -> bool:
    """Ask the user to confirm an irreversible action unless skipped explicitly."""
    if skip_confirmation:
        return True
    sys.stderr.write(prompt)
    sys.stderr.flush()
    answer = input().strip().lower()
    return answer in {"y", "yes"}


def get_config_dir(
    project_name: str,
    study_name: str | None,
    project_root: str | Path | None = None,
) -> Path:
    """Return the directory that stores optimization config files for the target scope."""
    project_dir = get_project_dir(project_name, project_root)
    if study_name is None:
        return project_dir
    return get_study_dir(project_dir, study_name)


def validate_project_configs(
    project_dir: Path,
    config_dir: Path,
    study_name: str | None,
) -> dict:
    """Validate the three project/study config files and return JSON-serializable results."""
    checks = (
        (BaseConfig.OPTIMIZATION_CONFIG_FILENAME.value, OptimizationConfig),
        (BaseConfig.MACHINE_CONFIG_FILENAME.value, MachineConfig),
        (BaseConfig.OPTIMIZATION_PROBLEM_CONFIG_FILENAME.value, OptimizationProblemConfig),
    )
    errors = []
    for filename, model in checks:
        file_result = _validate_config_file(config_dir / filename, filename, model)
        if file_result is not None:
            errors.append(file_result)
    return {
        "project": project_dir.name,
        "study": study_name,
        "config_dir": str(config_dir),
        "valid": len(errors) == 0,
        "errors": errors,
    }


def _validate_config_file(config_path: Path, filename: str, model: type[BaseModel]) -> dict | None:
    result = None
    if not config_path.exists():
        result = {
            "file": filename,
            "path": str(config_path),
            "status": "missing",
            "errors": [{"message": "Config file does not exist."}],
        }
    else:
        try:
            with config_path.open(encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            result = {
                "file": filename,
                "path": str(config_path),
                "status": "invalid_yaml",
                "errors": [{"message": str(e)}],
            }
        except OSError as e:
            result = {
                "file": filename,
                "path": str(config_path),
                "status": "read_error",
                "errors": [{"message": str(e)}],
            }
        else:
            if data is None:
                result = {
                    "file": filename,
                    "path": str(config_path),
                    "status": "empty",
                    "errors": [{"message": "Config file is empty."}],
                }
            elif not isinstance(data, dict):
                result = {
                    "file": filename,
                    "path": str(config_path),
                    "status": "invalid",
                    "errors": [{"message": "Config root must be a mapping."}],
                }
            else:
                errors = _validate_config_data(data, model)
                if errors:
                    result = _config_file_result(config_path, filename, "invalid", errors)
    return result


def _config_file_result(config_path: Path, filename: str, status: str, errors: list[dict]) -> dict:
    return {
        "file": filename,
        "path": str(config_path),
        "status": status,
        "errors": errors,
    }


def _collect_non_string_root_key_errors(data: dict) -> list[dict]:
    return [
        {
            "location": [str(key)],
            "message": f"Config root keys must be strings, got {type(key).__name__}: {key!r}",
            "type": "invalid_mapping_key",
        }
        for key in data
        if not isinstance(key, str)
    ]


def _format_pydantic_errors(error: ValidationError) -> list[dict]:
    return [
        {
            "location": [str(part) for part in item.get("loc", ())],
            "message": item.get("msg", ""),
            "type": item.get("type", ""),
        }
        for item in error.errors()
    ]


def _validate_config_data(data: dict, model: type[BaseModel]) -> list[dict]:
    non_string_key_errors = _collect_non_string_root_key_errors(data)
    try:
        model.model_validate(data)
    except ValidationError as e:
        return [*non_string_key_errors, *_format_pydantic_errors(e)]
    except (TypeError, ValueError) as e:
        return [
            *non_string_key_errors,
            {
                "location": [],
                "message": str(e),
                "type": type(e).__name__,
            },
        ]
    return non_string_key_errors


def ensure_study_from_source(
    project_name: str,
    source_dir: Path,
    study_name: str,
    project_root: str | Path | None = None,
) -> Path:
    """Create or refresh a study by copying config files from a source directory."""
    project_dir = get_project_dir(project_name, project_root)
    studies_dir = project_dir / BaseConfig.OPTIMIZATION_STUDIES_DIRNAME.value
    studies_dir.mkdir(exist_ok=True)
    study_dir = studies_dir / study_name
    study_dir.mkdir(exist_ok=True)
    for config_name in (
        BaseConfig.OPTIMIZATION_CONFIG_FILENAME.value,
        BaseConfig.MACHINE_CONFIG_FILENAME.value,
        BaseConfig.OPTIMIZATION_PROBLEM_CONFIG_FILENAME.value,
    ):
        src_file = source_dir / config_name
        if src_file.exists():
            shutil.copy(src_file, study_dir / config_name)
    return study_dir


def ensure_default_study(
    project_name: str,
    project_root: str | Path | None = None,
) -> Path:
    """Ensure the default study exists by copying root config files on first use."""
    project_dir = get_project_dir(project_name, project_root)
    return ensure_study_from_source(project_name, project_dir, DEFAULT_STUDY_NAME, project_root)


def resolve_effective_study_name(
    project_name: str,
    study_name: str | None,
    project_root: str | Path | None = None,
) -> str:
    """Resolve the effective study name, creating the default study when needed."""
    if study_name is not None:
        return study_name
    ensure_default_study(project_name, project_root)
    return DEFAULT_STUDY_NAME


def load_optimization_yaml(
    project_name: str,
    study_name: str,
    project_root: str | Path | None = None,
) -> dict:
    """Load optimization.yaml from a study."""
    optimization_filepath = (
        get_config_dir(project_name, study_name, project_root) / BaseConfig.OPTIMIZATION_CONFIG_FILENAME.value
    )
    with optimization_filepath.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_restart_target_study_name(
    project_name: str,
    source_study_name: str,
    project_root: str | Path | None = None,
) -> str:
    """Resolve the restart target study from the source study config."""
    optimization_config = load_optimization_yaml(
        project_name,
        source_study_name,
        project_root,
    )
    restart_config = optimization_config.get("restart", {}) or {}
    target_study_name = restart_config.get("target_study")
    if isinstance(target_study_name, str) and target_study_name.strip():
        return target_study_name.strip()
    return f"{source_study_name}_restart"


def ensure_restart_target_study(
    project_name: str,
    source_study_name: str,
    target_study_name: str,
    project_root: str | Path | None = None,
) -> Path:
    """Ensure the restart target study exists, cloning from source when no explicit target is configured."""
    source_dir = get_config_dir(project_name, source_study_name, project_root)
    optimization_config = load_optimization_yaml(
        project_name,
        source_study_name,
        project_root,
    )
    restart_config = optimization_config.get("restart", {}) or {}
    configured_target = restart_config.get("target_study")
    target_dir = (
        get_project_dir(project_name, project_root) / BaseConfig.OPTIMIZATION_STUDIES_DIRNAME.value / target_study_name
    )
    if isinstance(configured_target, str) and configured_target.strip():
        if not target_dir.exists():
            msg = f"Restart target study '{target_study_name}' does not exist in project '{project_name}'."
            raise ValueError(msg)
        return target_dir
    return ensure_study_from_source(project_name, source_dir, target_study_name, project_root)


def inject_restart_mean(optimization_config: dict, mean: list[float]) -> dict:
    """Inject a restart mean vector into optimizer kwargs."""
    optimizer_config = optimization_config.get("optimizer", {}) or {}
    optimizer_name = optimizer_config.get("name")
    kwargs = dict(optimizer_config.get("kwargs", {}) or {})
    kwargs["mean"] = mean
    optimization_config["optimizer"] = {"name": optimizer_name, "kwargs": kwargs}
    return optimization_config


def apply_restart_selection(
    project_name: str,
    source_study_name: str,
    selection: RestartSelection,
    project_root: str | Path | None = None,
) -> str:
    """Prepare a restart target study and inject the selected mean vector."""
    target_study_name = selection.target_study_name
    study_dir = ensure_restart_target_study(
        project_name,
        source_study_name,
        target_study_name,
        project_root,
    )
    optimization_filepath = study_dir / BaseConfig.OPTIMIZATION_CONFIG_FILENAME.value
    with optimization_filepath.open(encoding="utf-8") as f:
        optimization_config: dict = yaml.safe_load(f) or {}
    optimization_config = inject_restart_mean(optimization_config, selection.mean)
    optimization_config["resource_dir"] = None
    optimization_config["output_dir"] = None
    with optimization_filepath.open("w", encoding="utf-8") as f:
        yaml.safe_dump(optimization_config, f, sort_keys=False)
    return target_study_name


def get_summary_dir(
    project_name: str,
    study_name: str,
    project_root: str | Path | None = None,
) -> Path:
    """Return the summary directory for a study-level run."""
    project_dir = get_project_dir(project_name, project_root)
    return project_dir / BaseConfig.SUMMARY_DIRNAME.value / BaseConfig.OPTIMIZATION_STUDIES_DIRNAME.value / study_name


def get_study_runs_dir(
    project_name: str,
    study_name: str,
    project_root: str | Path | None = None,
) -> Path:
    """Return the runs directory for the target study."""
    return get_summary_dir(project_name, study_name, project_root) / BaseConfig.RUNS_DIRNAME.value


def allocate_run_id(
    project_name: str,
    study_name: str,
    project_root: str | Path | None = None,
) -> str:
    """Allocate the next sequential run id under a study summary."""
    runs_dir = get_study_runs_dir(project_name, study_name, project_root)
    runs_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r"^run_(\d+)$")
    max_seq = 0
    for child in runs_dir.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if match is None:
            continue
        max_seq = max(max_seq, int(match.group(1)))
    return f"run_{max_seq + 1:04d}"


def validate_run_id(run_id: str) -> str:
    """Validate a canonical run id generated by allocate_run_id."""
    if not isinstance(run_id, str) or not run_id.startswith("run_"):
        raise ValueError(_invalid_run_id_message(run_id))
    seq_text = run_id.removeprefix("run_")
    if not seq_text.isdecimal():
        raise ValueError(_invalid_run_id_message(run_id))
    seq = int(seq_text)
    if seq < 1 or run_id != f"run_{seq:04d}":
        raise ValueError(_invalid_run_id_message(run_id))
    return run_id


def _invalid_run_id_message(run_id: object) -> str:
    return (
        f"Invalid run_id '{run_id}'. Expected canonical run id format 'run_{{n:04d}}' with n >= 1 "
        "(for example, 'run_0001'). Numeric shorthand, extra zero padding, and path separators are not supported."
    )


def get_run_summary_dir(
    project_name: str,
    study_name: str,
    run_id: str,
    project_root: str | Path | None = None,
) -> Path:
    """Return the summary directory for a specific study run."""
    return get_study_runs_dir(project_name, study_name, project_root) / run_id


def get_existing_run_summary_dir(
    project_name: str,
    study_name: str,
    run_id: str,
    project_root: str | Path | None = None,
) -> Path:
    """Resolve an existing run summary directory after validating run_id."""
    validated_run_id = validate_run_id(run_id)
    runs_dir = get_study_runs_dir(project_name, study_name, project_root)
    run_summary_dir = runs_dir / validated_run_id
    if not run_summary_dir.is_dir():
        msg = (
            f"Run summary directory not found for project_name='{project_name}', study_name='{study_name}', "
            f"run_id='{validated_run_id}', runs_dir='{runs_dir}'."
        )
        raise FileNotFoundError(msg)
    try:
        resolved_runs_dir = runs_dir.resolve(strict=True)
        resolved_run_summary_dir = run_summary_dir.resolve(strict=True)
    except OSError as e:
        msg = (
            f"Failed to resolve run directory for project_name='{project_name}', study_name='{study_name}', "
            f"run_id='{validated_run_id}', runs_dir='{runs_dir}': {e}"
        )
        raise ValueError(msg) from e
    expected_run_summary_dir = resolved_runs_dir / validated_run_id
    if resolved_run_summary_dir != expected_run_summary_dir:
        msg = (
            f"Invalid run directory for project_name='{project_name}', study_name='{study_name}', "
            f"run_id='{validated_run_id}', runs_dir='{runs_dir}'."
        )
        raise ValueError(msg)
    return run_summary_dir


def write_latest_run_info(
    study_summary_dir: Path,
    run_id: str,
) -> None:
    """Persist metadata that points to the latest run of a study."""
    study_summary_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "latest_run_id": run_id,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    with (study_summary_dir / BaseConfig.LATEST_RUN_FILENAME.value).open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def sync_latest_study_outputs(
    study_summary_dir: Path,
    run_summary_dir: Path,
    run_id: str,
) -> None:
    """Refresh the latest-result shortcuts for a study after a run finishes."""
    write_latest_run_info(study_summary_dir, run_id)
    for dirname in (BaseConfig.RESULTANT_DIRNAME.value, BaseConfig.RECORDS_DIRNAME.value):
        dst_dir = study_summary_dir / dirname
        src_dir = run_summary_dir / dirname
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        if src_dir.exists():
            shutil.copytree(src_dir, dst_dir)


def get_latest_run_summary_dir(
    project_name: str,
    study_name: str,
    project_root: str | Path | None = None,
) -> Path:
    """Resolve the latest run directory for a study."""
    project_dir = get_project_dir(project_name, project_root)
    legacy_summary_dir = project_dir / BaseConfig.SUMMARY_DIRNAME.value
    study_summary_dir = get_summary_dir(project_name, study_name, project_root)
    latest_run_info = study_summary_dir / BaseConfig.LATEST_RUN_FILENAME.value
    if not latest_run_info.exists():
        legacy_resultant_dir = legacy_summary_dir / BaseConfig.RESULTANT_DIRNAME.value
        if legacy_resultant_dir.exists():
            logger.info(
                "No latest run info found for study %s in project %s. Falling back to legacy summary directory %s.",
                study_name,
                project_name,
                legacy_summary_dir,
            )
            return legacy_summary_dir
        msg = f"No latest run info found for study {study_name} in project {project_name}."
        raise FileNotFoundError(msg)
    with latest_run_info.open(encoding="utf-8") as f:
        payload: dict | None = yaml.safe_load(f)
    if not isinstance(payload, dict) or "latest_run_id" not in payload:
        msg = f"Invalid latest run info in {latest_run_info}."
        raise ValueError(msg)
    return get_run_summary_dir(project_name, study_name, str(payload["latest_run_id"]), project_root)


def finalize_study_run(
    project_name: str,
    study_name: str,
    run_summary_dir: Path,
    run_id: str,
    project_root: str | Path | None = None,
) -> None:
    """Generate study-level artifacts after a run finishes."""
    logger.info("Finalizing run...")
    project_dir = get_project_dir(project_name, project_root)
    summary_dir = get_summary_dir(project_name, study_name, project_root)
    study_result_dir = run_summary_dir / BaseConfig.RESULTANT_DIRNAME.value
    study_records_csv = run_summary_dir / BaseConfig.RECORDS_DIRNAME.value / "run_records.csv"
    output_surrogate_result_artifacts(
        str(study_result_dir),
        source_csv=study_records_csv,
        metadata={
            "source_study_ids": [study_name],
            "run_id": run_id,
            "filter_conditions": {
                "study_ids": [study_name],
                "statuses": ["success", "failed"],
            },
        },
    )
    append_cross_study_table(
        str(project_dir / BaseConfig.SUMMARY_DIRNAME.value),
        str(run_summary_dir),
        project_name=project_name,
        study_name=study_name,
        run_id=run_id,
    )
    sync_latest_study_outputs(summary_dir, run_summary_dir, str(run_id))
    logger.info("Finalizing run finished.")


def analyze_runs_for_study(
    *,
    project_name: str,
    study_name: str | None,
    reference_point: list[float] | None = None,
    run_ids: list[str] | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Create comparison/statistics artifacts for all runs under a study."""
    current_study_name = resolve_effective_study_name(project_name, study_name, project_root)
    project_summary_dir = get_project_dir(project_name, project_root) / BaseConfig.SUMMARY_DIRNAME.value
    return analyze_study_runs(
        project_summary_dir,
        study_name=current_study_name,
        runs_dirname=BaseConfig.RUNS_DIRNAME.value,
        resultant_dirname=BaseConfig.RESULTANT_DIRNAME.value,
        reference_point=reference_point,
        run_ids=run_ids,
    )


def build_analysis_visualizer_data(
    *,
    project_name: str,
    study_name: str,
    run_ids: list[str] | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build GUI-ready multi-run data from the study summary."""
    project_summary_dir = get_project_dir(project_name, project_root) / BaseConfig.SUMMARY_DIRNAME.value
    return build_multi_run_visualizer_data(
        project_summary_dir,
        study_name=study_name,
        runs_dirname=BaseConfig.RUNS_DIRNAME.value,
        resultant_dirname=BaseConfig.RESULTANT_DIRNAME.value,
        run_ids=run_ids,
    )
