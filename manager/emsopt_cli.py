"""
emsopt_cli.py
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

import contextlib
import logging
import queue
import shutil
import threading
from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path

import yaml

import utils.emsopt_cli_helpers as cli_helpers
from emsopt_engine.configs.base import BaseConfig
from emsopt_engine.configs.machine import MachineConfig
from emsopt_engine.setup_project import (
    get_available_config_detail,
    get_available_config_options,
    get_study_dir,
)
from manager.background_job import run_background_job_cli
from manager.emsopt_api import EMSOptimizerClient
from manager.optimization_output_handler import (
    export_surrogate_training_data,
)
from manager.optimization_visualizer import OptimizationVisualizer, VisualizerPayload
from manager.run_analysis import ANALYSIS_DIRNAME

logging.basicConfig(level=logging.INFO)
DEFAULT_STUDY_NAME = cli_helpers.DEFAULT_STUDY_NAME


def _add_project_root_argument(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Parent directory that contains EMSOptimizer projects. Defaults to EMSOptimizer's project dir.",
    )


def setup_argument_parser() -> ArgumentParser:  # noqa: PLR0915
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(required=True, dest="command")

    # pre-process
    parser_pre = subparsers.add_parser(
        "show_avl",
        help="""
        This command shows available options
        for optimizer, evaluator, level_set_function, and optimization problem function.
        If -n or --name option is specified, the detail of the available will be displayed
        """,
    )
    parser_pre.add_argument("-n", "--name", type=str, default=None, help="Name of the availables to be displayed")
    parser_pre.set_defaults(func=show_avl)

    parser_pre = subparsers.add_parser("cp_proj", help="This command copies a project.")
    parser_pre.add_argument("src_project_name", type=str, help="Name of source project.")
    parser_pre.add_argument("dst_project_name", type=str, help="Name of destination new project.")
    _add_project_root_argument(parser_pre)
    parser_pre.set_defaults(func=cp_proj)

    parser_pre = subparsers.add_parser(
        "cln_proj", help="This command clean up a project by removing intermediate and result dirs."
    )
    parser_pre.add_argument("project_name", type=str, help="Name of project.")
    _add_project_root_argument(parser_pre)
    parser_pre.add_argument(
        "--remove-studies",
        action="store_true",
        help="Also remove all optimization study directories in the project.",
    )
    parser_pre.add_argument(
        "--remove-summary",
        action="store_true",
        help="Also remove the project summary directory.",
    )
    parser_pre.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt and execute the cleanup immediately.",
    )
    parser_pre.set_defaults(func=cln_proj)

    parser_pre = subparsers.add_parser("rm_proj", help="This command removes a project.")
    parser_pre.add_argument("project_name", type=str, help="Name of project.")
    _add_project_root_argument(parser_pre)
    parser_pre.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt and execute the cleanup immediately.",
    )
    parser_pre.set_defaults(func=rm_proj)

    parser_pre = subparsers.add_parser(
        "save_tpl", help="This command saves current optimization_problem config as template."
    )
    parser_pre.add_argument("project_name", type=str, help="Name of the project.")
    parser_pre.add_argument("tpl_name", type=str, help="Name of template.")
    parser_pre.add_argument("--study-name", type=str, default=None, help="Optimization study name.")
    _add_project_root_argument(parser_pre)
    parser_pre.set_defaults(func=save_tpl)

    parser_pre = subparsers.add_parser("load_tpl", help="This command loads optimization_problem config from template.")
    parser_pre.add_argument("project_name", type=str, help="Name of the project.")
    parser_pre.add_argument("tpl_name", type=str, help="Name of template to load.")
    parser_pre.add_argument("--study-name", type=str, default=None, help="Optimization study name.")
    _add_project_root_argument(parser_pre)
    parser_pre.set_defaults(func=load_tpl)

    parser_pre = subparsers.add_parser("mk_study", help="This command creates an optimization study in a project.")
    parser_pre.add_argument("project_name", type=str, help="Name of the project.")
    parser_pre.add_argument("study_name", type=str, help="Name of the optimization study to create.")
    parser_pre.add_argument("--from-study", type=str, default=None, help="Source optimization study name.")
    _add_project_root_argument(parser_pre)
    parser_pre.set_defaults(func=mk_study)

    parser_pre = subparsers.add_parser(
        "validate", help="This command validates optimization, machine, and optimization_problem configs as JSON."
    )
    parser_pre.add_argument("project_name", type=str, help="Name of the project.")
    parser_pre.add_argument("--study-name", type=str, default=None, help="Optimization study name.")
    _add_project_root_argument(parser_pre)
    parser_pre.set_defaults(func=validate)

    parser_pre = subparsers.add_parser(
        "inspect_mesh",
        help="Inspect physical IDs, element types, node counts, and polar ranges in a gmsh mesh.",
    )
    parser_pre.add_argument("mesh", type=str, help="Path to the gmsh mesh file.")
    parser_pre.add_argument(
        "--machine-config",
        type=str,
        default=None,
        help="Optional machine.yaml path used for material-ID consistency checks.",
    )
    parser_pre.set_defaults(func=inspect_mesh)

    parser_pre = subparsers.add_parser(
        "validate_mesh",
        help="Run lightweight project/study checks before optimization, including mesh/material consistency.",
    )
    parser_pre.add_argument("project_name", type=str, help="Name of the project.")
    parser_pre.add_argument("--study-name", type=str, default=None, help="Optimization study name.")
    parser_pre.add_argument(
        "--mesh",
        type=str,
        default=None,
        help="Optional mesh path. Defaults to machine.yaml design target mesh in the project/study.",
    )
    _add_project_root_argument(parser_pre)
    parser_pre.set_defaults(func=validate_mesh)

    # execution-process
    parser_run = subparsers.add_parser("run", help="This command runs optimization based on project's configurations.")
    parser_run.add_argument("project_name", type=str, help="Name of the project.")
    parser_run.add_argument("--study-name", type=str, default=None, help="Optimization study name.")
    _add_project_root_argument(parser_run)
    parser_run.add_argument(
        "--background",
        action="store_true",
        help="Start the optimization in a background process and return immediately.",
    )
    parser_run.set_defaults(func=run)

    parser_run = subparsers.add_parser(
        "batch_run",
        help="This command runs the same optimization study multiple times as independent runs.",
    )
    parser_run.add_argument("project_name", type=str, help="Name of the project.")
    parser_run.add_argument("num_runs", type=int, help="Number of independent runs.")
    parser_run.add_argument("--study-name", type=str, default=None, help="Optimization study name.")
    _add_project_root_argument(parser_run)
    parser_run.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the batch immediately when a run fails.",
    )
    parser_run.add_argument(
        "--background",
        action="store_true",
        help="Start the batch in a background process and return immediately.",
    )
    parser_run.set_defaults(func=batch_run)

    parser_run = subparsers.add_parser("sample", help="This command sample individuals of project and evaluate them.")
    parser_run.add_argument("project_name", type=str, help="Name of the project.")
    parser_run.add_argument("num_sample", type=int, help="Number of sample.")
    parser_run.add_argument("--study-name", type=str, default=None, help="Optimization study name.")
    parser_run.add_argument("--chunk-size", type=int, default=None, help="Number of samples evaluated per chunk.")
    parser_run.add_argument(
        "--background",
        action="store_true",
        help="Start the sample evaluation in a background process and return immediately.",
    )
    _add_project_root_argument(parser_run)
    parser_run.set_defaults(func=sample)

    # job-related-process
    parser_job = subparsers.add_parser("job_list", help="List background optimization jobs for a study.")
    parser_job.add_argument("project_name", type=str, help="Name of the project.")
    parser_job.add_argument("--study-name", type=str, default=None, help="Optimization study name.")
    _add_project_root_argument(parser_job)
    parser_job.set_defaults(func=job_list)

    parser_job = subparsers.add_parser("job_status", help="Show background optimization job status.")
    parser_job.add_argument("project_name", type=str, help="Name of the project.")
    parser_job.add_argument("--study-name", type=str, default=None, help="Optimization study name.")
    parser_job.add_argument("--job-id", required=True, type=str, help="Background job id.")
    _add_project_root_argument(parser_job)
    parser_job.set_defaults(func=job_status)

    parser_job = subparsers.add_parser("job_pause", help="Pause a background optimization job.")
    parser_job.add_argument("project_name", type=str, help="Name of the project.")
    parser_job.add_argument("--study-name", type=str, default=None, help="Optimization study name.")
    parser_job.add_argument("--job-id", required=True, type=str, help="Background job id.")
    _add_project_root_argument(parser_job)
    parser_job.set_defaults(func=job_pause)

    parser_job = subparsers.add_parser("job_resume", help="Resume a paused background optimization job.")
    parser_job.add_argument("project_name", type=str, help="Name of the project.")
    parser_job.add_argument("--study-name", type=str, default=None, help="Optimization study name.")
    parser_job.add_argument("--job-id", required=True, type=str, help="Background job id.")
    _add_project_root_argument(parser_job)
    parser_job.set_defaults(func=job_resume)

    parser_job = subparsers.add_parser("job_stop", help="Stop a background optimization job.")
    parser_job.add_argument("project_name", type=str, help="Name of the project.")
    parser_job.add_argument("--study-name", type=str, default=None, help="Optimization study name.")
    parser_job.add_argument("--job-id", required=True, type=str, help="Background job id.")
    _add_project_root_argument(parser_job)
    parser_job.set_defaults(func=job_stop)

    parser_worker = subparsers.add_parser("_run_background_job", help="Internal background job worker.")
    parser_worker.add_argument("job_dir", type=str, help="Background job directory.")
    parser_worker.set_defaults(func=_run_background_job)

    # post-process
    parser_post = subparsers.add_parser("check", help="This command checks performed optimization summary.")
    parser_post.add_argument("project_name", type=str, help="Name of the project.")
    parser_post.add_argument("--study-name", type=str, default=None, help="Optimization study name.")
    parser_post.add_argument("--run-id", type=str, default=None, help="Run ID to check, e.g. run_0001.")
    _add_project_root_argument(parser_post)
    parser_post.set_defaults(func=check)

    parser_post = subparsers.add_parser(
        "export_data",
        help="This command exports surrogate training data from the project-level cross-study table.",
    )
    parser_post.add_argument("project_name", type=str, help="Name of the project.")
    _add_project_root_argument(parser_post)
    parser_post.add_argument(
        "--study-names",
        nargs="+",
        default=None,
        help="Target optimization study names. If omitted, all studies in the project are used.",
    )
    parser_post.add_argument(
        "--statuses",
        nargs="+",
        default=["success"],
        help="Record statuses to export. Defaults to success only.",
    )
    parser_post.set_defaults(func=export_data)

    parser_post = subparsers.add_parser(
        "analyze_runs",
        help="This command builds multi-run comparison plots and statistics for one study.",
    )
    parser_post.add_argument("project_name", type=str, help="Name of the project.")
    parser_post.add_argument("--study-name", type=str, default=None, help="Optimization study name.")
    _add_project_root_argument(parser_post)
    parser_post.add_argument(
        "--reference-point",
        nargs="+",
        type=float,
        default=None,
        help="Reference point used for multi-objective hypervolume.",
    )
    parser_post.add_argument(
        "--run-ids",
        nargs="+",
        type=str,
        default=None,
        help="Optional subset of run ids to analyze.",
    )
    parser_post.add_argument(
        "--gui",
        action="store_true",
        help="Open the existing GUI in multi-run analysis mode after generating artifacts.",
    )
    parser_post.set_defaults(func=analyze_runs)

    return parser


def _validate_project(
    project_dir: Path,
    study_name: str | None = None,
    *,
    validate_configs: bool = False,
) -> dict | None:
    """Validate existence of a project directory and optionally validate its config files."""
    if not project_dir.exists():
        msg = f"Project name {project_dir} doesn't exists."
        raise FileNotFoundError(msg)
    if not validate_configs:
        return None

    config_dir = project_dir if study_name is None else get_study_dir(project_dir, study_name)
    return cli_helpers.validate_project_configs(project_dir, config_dir, study_name)


def cp_proj(
    src_project_name: str,
    dst_project_name: str,
    project_root: str | None = None,
    command: str | None = None,
    **_: object,
) -> None:
    src_dir = cli_helpers.get_project_dir(src_project_name, project_root)
    _validate_project(src_dir)
    dst_dir = cli_helpers.get_project_dir(dst_project_name, project_root)
    if dst_dir.exists():
        msg = f"Project folder {dst_dir} already exists."
        raise FileExistsError(msg)
    shutil.copytree(src_dir, dst_dir)
    cli_helpers.emit_command_result(
        command,
        {
            "status": "completed",
            "src_project": src_project_name,
            "dst_project": dst_project_name,
            "dst_dir": str(dst_dir),
        },
    )


def cln_proj(
    project_name: str,
    *,
    remove_studies: bool = False,
    remove_summary: bool = False,
    yes: bool = False,
    project_root: str | None = None,
    command: str | None = None,
    **_: object,
) -> None:
    project_dir = cli_helpers.get_project_dir(project_name, project_root)
    _validate_project(project_dir)

    targets = [
        project_dir / BaseConfig.RESOURCE_DIRNAME.value,
        project_dir / BaseConfig.PROGRESS_DIRNAME.value,
    ]
    if remove_summary:
        targets.append(project_dir / BaseConfig.SUMMARY_DIRNAME.value)
    studies_dir = project_dir / BaseConfig.OPTIMIZATION_STUDIES_DIRNAME.value
    if studies_dir.exists():
        for study_dir in studies_dir.iterdir():
            if not study_dir.is_dir():
                continue
            targets.extend(
                [
                    study_dir / BaseConfig.RESOURCE_DIRNAME.value,
                    study_dir / BaseConfig.PROGRESS_DIRNAME.value,
                ]
            )
        if remove_studies:
            targets.append(studies_dir)

    unique_targets = []
    seen_targets = set()
    for target in targets:
        if target in seen_targets:
            continue
        seen_targets.add(target)
        unique_targets.append(target)

    action_summary = [str(target.relative_to(project_dir)) for target in unique_targets]
    if not cli_helpers.confirm_irreversible_action(
        f"Clean project '{project_name}' by removing: {', '.join(action_summary)} ? [y/N]: ",
        skip_confirmation=yes,
    ):
        cli_helpers.emit_command_result(
            command,
            {
                "status": "cancelled",
                "project": project_name,
                "removed": [],
            },
        )
        return

    removed = []
    for target in unique_targets:
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(target)
            removed.append(str(target))
    cli_helpers.emit_command_result(
        command,
        {
            "status": "completed",
            "project": project_name,
            "removed": removed,
        },
    )


def rm_proj(
    project_name: str,
    *,
    yes: bool = False,
    project_root: str | None = None,
    command: str | None = None,
    **_: object,
) -> None:
    if not cli_helpers.confirm_irreversible_action(
        f"Remove project '{project_name}'? [y/N]: ",
        skip_confirmation=yes,
    ):
        cli_helpers.emit_command_result(
            command,
            {
                "status": "cancelled",
                "project": project_name,
            },
        )
        return
    shutil.rmtree(cli_helpers.get_project_dir(project_name, project_root))
    cli_helpers.emit_command_result(
        command,
        {
            "status": "completed",
            "project": project_name,
        },
    )


def save_tpl(
    project_name: str,
    tpl_name: str,
    study_name: str | None = None,
    project_root: str | None = None,
    command: str | None = None,
    **_: object,
) -> None:
    template_filepath = cli_helpers.get_project_root(project_root) / BaseConfig.PROBLEM_TEMPLATE_FILENAME.value
    project_config_filepath = (
        cli_helpers.get_config_dir(project_name, study_name, project_root)
        / BaseConfig.OPTIMIZATION_PROBLEM_CONFIG_FILENAME.value
    )

    if template_filepath.exists():
        with template_filepath.open(encoding="utf-8") as f:
            templates = yaml.safe_load(f)
        if templates is None:
            templates = {}
    else:
        templates = {}

    with project_config_filepath.open(encoding="utf-8") as f:
        new_template = yaml.safe_load(f)
    if tpl_name in templates:
        msg = f"Template name {tpl_name} already exists."
        raise ValueError(msg)
    templates[tpl_name] = new_template

    with template_filepath.open("w", encoding="utf-8") as f:
        yaml.safe_dump(templates, f)
    cli_helpers.emit_command_result(
        command,
        {
            "status": "completed",
            "project": project_name,
            "study": study_name,
            "template": tpl_name,
            "template_file": str(template_filepath),
        },
    )


def load_tpl(
    project_name: str,
    tpl_name: str,
    study_name: str | None = None,
    project_root: str | None = None,
    command: str | None = None,
    **_: object,
) -> None:
    template_filepath = cli_helpers.get_project_root(project_root) / BaseConfig.PROBLEM_TEMPLATE_FILENAME.value
    project_config_filepath = (
        cli_helpers.get_config_dir(project_name, study_name, project_root)
        / BaseConfig.OPTIMIZATION_PROBLEM_CONFIG_FILENAME.value
    )

    if template_filepath.exists():
        with template_filepath.open(encoding="utf-8") as f:
            templates = yaml.safe_load(f)
        if templates is None:
            templates = {}
    else:
        templates = {}

    if tpl_name not in templates:
        msg = f"Template name {tpl_name} doesn't exists."
        raise ValueError(msg)
    with project_config_filepath.open("w", encoding="utf-8") as f:
        yaml.safe_dump(templates[tpl_name], f)
    cli_helpers.emit_command_result(
        command,
        {
            "status": "completed",
            "project": project_name,
            "study": study_name,
            "template": tpl_name,
            "template_file": str(template_filepath),
            "config_file": str(project_config_filepath),
        },
    )


def mk_study(
    project_name: str,
    study_name: str,
    from_study: str | None = None,
    project_root: str | None = None,
    command: str | None = None,
    **_: object,
) -> None:
    """Create a new optimization study by copying config files from the project root or another study."""
    project_dir = cli_helpers.get_project_dir(project_name, project_root)
    _validate_project(project_dir)
    studies_dir = project_dir / BaseConfig.OPTIMIZATION_STUDIES_DIRNAME.value
    studies_dir.mkdir(exist_ok=True)
    dst_study_dir = studies_dir / study_name
    if dst_study_dir.exists():
        msg = f"Optimization study folder {dst_study_dir} already exists."
        raise FileExistsError(msg)

    src_dir = project_dir if from_study is None else get_study_dir(project_dir, from_study)
    cli_helpers.ensure_study_from_source(project_name, src_dir, study_name, project_root)
    cli_helpers.emit_command_result(
        command,
        {
            "status": "completed",
            "project": project_name,
            "study": study_name,
            "from_study": from_study,
            "study_dir": str(dst_study_dir),
        },
    )


def _create_client(project_root: str | None = None) -> EMSOptimizerClient:
    return EMSOptimizerClient(project_root=project_root)


def run(
    project_name: str,
    *,
    study_name: str | None = None,
    project_root: str | None = None,
    background: bool = False,
    command: str | None = None,
    **_: object,
) -> None:
    project_dir = cli_helpers.get_project_dir(project_name, project_root)
    _validate_project(project_dir)
    client = _create_client(project_root)
    if background:
        # Background mode only launches the worker; completion/failure is observed via job_* commands.
        result = client.start_background_run(project_name, study_name)
        cli_helpers.emit_command_result(
            command,
            {
                **result,
                "project": project_name,
                "study": study_name,
            },
        )
        return
    client.run(project_name, study_name)
    cli_helpers.emit_command_result(
        command,
        {
            "status": "completed",
            "project": project_name,
            "study": study_name,
        },
    )


def check(
    project_name: str,
    study_name: str | None = None,
    project_root: str | None = None,
    command: str | None = None,
    *,
    run_id: str | None = None,
    **_: object,
) -> None:
    project_dir = cli_helpers.get_project_dir(project_name, project_root)
    _validate_project(project_dir)
    result = _create_client(project_root).check(project_name, study_name, run_id=run_id)
    cli_helpers.emit_command_result(
        command,
        {
            "status": "completed",
            "project": project_name,
            "study": study_name,
            "run_id": result.run_id,
        },
    )


def batch_run(
    project_name: str,
    num_runs: int,
    *,
    study_name: str | None = None,
    stop_on_error: bool = False,
    project_root: str | None = None,
    background: bool = False,
    command: str | None = None,
    **_: object,
) -> None:
    project_dir = cli_helpers.get_project_dir(project_name, project_root)
    _validate_project(project_dir)
    client = _create_client(project_root)
    if background:
        # Store batch parameters in job_info.yaml so the detached worker can replay the request.
        result = client.start_background_batch_run(
            project_name,
            num_runs,
            study_name,
            stop_on_error=stop_on_error,
        )
        cli_helpers.emit_command_result(
            command,
            {
                **result,
                "project": project_name,
                "study": study_name,
                "num_runs": num_runs,
            },
        )
        return
    client.batch_run(project_name, num_runs, study_name, stop_on_error=stop_on_error)
    cli_helpers.emit_command_result(
        command,
        {
            "status": "completed",
            "project": project_name,
            "study": study_name,
            "num_runs": num_runs,
        },
    )


def job_list(
    project_name: str,
    study_name: str | None = None,
    project_root: str | None = None,
    command: str | None = None,
    **_: object,
) -> None:
    """Validate the project and emit all background jobs for the target study."""
    project_dir = cli_helpers.get_project_dir(project_name, project_root)
    _validate_project(project_dir)
    jobs = _create_client(project_root).list_background_jobs(project_name, study_name)
    cli_helpers.emit_command_result(
        command,
        {
            "status": "completed",
            "project": project_name,
            "study": study_name,
            "jobs": jobs,
        },
    )


def job_status(
    project_name: str,
    job_id: str,
    study_name: str | None = None,
    project_root: str | None = None,
    command: str | None = None,
    **_: object,
) -> None:
    """Validate the project and emit one background job's persisted status."""
    project_dir = cli_helpers.get_project_dir(project_name, project_root)
    _validate_project(project_dir)
    result = _create_client(project_root).background_job_status(project_name, study_name, job_id)
    cli_helpers.emit_command_result(
        command,
        {
            "status": "completed",
            "project": project_name,
            "study": study_name,
            "job": result,
        },
    )


def job_pause(
    project_name: str,
    job_id: str,
    study_name: str | None = None,
    project_root: str | None = None,
    command: str | None = None,
    **_: object,
) -> None:
    """Request pause for a background job and emit the updated status payload."""
    project_dir = cli_helpers.get_project_dir(project_name, project_root)
    _validate_project(project_dir)
    result = _create_client(project_root).pause_background_job(project_name, study_name, job_id)
    cli_helpers.emit_command_result(
        command,
        {
            "status": "completed",
            "project": project_name,
            "study": study_name,
            "job": result,
        },
    )


def job_resume(
    project_name: str,
    job_id: str,
    study_name: str | None = None,
    project_root: str | None = None,
    command: str | None = None,
    **_: object,
) -> None:
    """Request resume for a background job and emit the updated status payload."""
    project_dir = cli_helpers.get_project_dir(project_name, project_root)
    _validate_project(project_dir)
    result = _create_client(project_root).resume_background_job(project_name, study_name, job_id)
    cli_helpers.emit_command_result(
        command,
        {
            "status": "completed",
            "project": project_name,
            "study": study_name,
            "job": result,
        },
    )


def job_stop(
    project_name: str,
    job_id: str,
    study_name: str | None = None,
    project_root: str | None = None,
    command: str | None = None,
    **_: object,
) -> None:
    """Request cooperative stop for a background job and emit the updated status."""
    project_dir = cli_helpers.get_project_dir(project_name, project_root)
    _validate_project(project_dir)
    result = _create_client(project_root).stop_background_job(project_name, study_name, job_id)
    cli_helpers.emit_command_result(
        command,
        {
            "status": "completed",
            "project": project_name,
            "study": study_name,
            "job": result,
        },
    )


def _run_background_job(job_dir: str, **_: object) -> None:
    """Run the internal worker entry point used by spawned background processes."""
    run_background_job_cli(job_dir)


def sample(
    project_name: str,
    num_sample: int,
    *,
    study_name: str | None = None,
    chunk_size: int | None = None,
    project_root: str | None = None,
    background: bool = False,
    command: str | None = None,
    **_: object,
) -> None:
    """Execute sample evaluation synchronously or start it as a background job."""
    project_dir = cli_helpers.get_project_dir(project_name, project_root)
    _validate_project(project_dir)
    client = _create_client(project_root)
    if background:
        # Store sample parameters in job_info.yaml so the detached worker can replay the request.
        result = client.start_background_sample(
            project_name,
            num_sample,
            study_name,
            chunk_size=chunk_size,
        )
        cli_helpers.emit_command_result(
            command,
            {
                **result,
                "project": project_name,
                "study": study_name,
                "num_sample": num_sample,
                "chunk_size": chunk_size,
            },
        )
        return
    client.sample(project_name, num_sample, study_name, chunk_size=chunk_size)
    cli_helpers.emit_command_result(
        command,
        {
            "status": "completed",
            "project": project_name,
            "study": study_name,
            "num_sample": num_sample,
            "chunk_size": chunk_size,
        },
    )


def analyze_runs(
    project_name: str,
    *,
    study_name: str | None = None,
    reference_point: list[float] | None = None,
    run_ids: list[str] | None = None,
    gui: bool = False,
    project_root: str | None = None,
    command: str | None = None,
    **_: object,
) -> None:
    project_dir = cli_helpers.get_project_dir(project_name, project_root)
    _validate_project(project_dir)

    analyze_kwargs = {
        "project_name": project_name,
        "study_name": study_name,
        "reference_point": reference_point,
        "run_ids": run_ids,
    }
    if project_root is not None:
        analyze_kwargs["project_root"] = project_root
    result = cli_helpers.analyze_runs_for_study(**analyze_kwargs)
    analysis_dir = Path(result["analysis_dir"]) if result.get("analysis_dir") else project_dir / ANALYSIS_DIRNAME
    if gui:
        if project_root is None:
            _show_analysis_gui(project_name, study_name, run_ids)
        else:
            _show_analysis_gui(project_name, study_name, run_ids, project_root)
    cli_helpers.emit_command_result(
        command,
        {
            "status": "completed",
            "project": project_name,
            "study": study_name,
            "analysis_dir": str(analysis_dir),
            "gui": gui,
        },
    )


def _show_analysis_gui(
    project_name: str,
    study_name: str | None,
    run_ids: list[str] | None,
    project_root: str | Path | None = None,
) -> None:
    """Open the existing visualizer in static multi-run analysis mode."""
    resolved_study_name = cli_helpers.resolve_effective_study_name(project_name, study_name, project_root)
    visualizer_data = cli_helpers.build_analysis_visualizer_data(
        project_name=project_name,
        study_name=resolved_study_name,
        run_ids=run_ids,
        project_root=project_root,
    )
    study_summary_dir = cli_helpers.get_summary_dir(project_name, resolved_study_name, project_root)
    progress_q: queue.Queue[VisualizerPayload] = queue.Queue()
    progress_q.put(
        VisualizerPayload(
            type="update",
            individuals=visualizer_data["individuals"],
            view_mode="analysis",
        )
    )
    visualizer = OptimizationVisualizer(str(study_summary_dir / ANALYSIS_DIRNAME))
    visualizer.run(
        visualizer_data["mode"],
        visualizer_data["num_obj"],
        progress_q,
        threading.Event(),
        threading.Event(),
        view_mode="analysis",
    )


def export_data(
    project_name: str,
    study_names: list[str] | None = None,
    statuses: list[str] | None = None,
    project_root: str | None = None,
    command: str | None = None,
    **_: object,
) -> None:
    """Export surrogate training data from the project-level cross-study table."""
    project_dir = cli_helpers.get_project_dir(project_name, project_root)
    _validate_project(project_dir)

    output_csv = export_surrogate_training_data(
        str(project_dir / BaseConfig.SUMMARY_DIRNAME.value),
        study_names=study_names,
        statuses=["success"] if statuses is None else statuses,
    )
    cli_helpers.emit_command_result(
        command,
        {
            "status": "not_found" if output_csv is None else "completed",
            "project": project_name,
            "study_names": study_names,
            "statuses": ["success"] if statuses is None else statuses,
            "output_csv": None if output_csv is None else str(output_csv),
        },
    )


def show_avl(name: str | None = None, command: str | None = None, **_: object) -> None:
    if name is None:
        cli_helpers.emit_command_result(
            command,
            {
                "status": "completed",
                "name": None,
                "found": True,
                "result": get_available_config_options(),
            },
        )
    else:
        msg = get_available_config_detail(name)
        if msg is None:
            cli_helpers.emit_command_result(
                command,
                {
                    "status": "not_found",
                    "name": name,
                    "found": False,
                    "result": None,
                },
            )
        else:
            cli_helpers.emit_command_result(
                command,
                {
                    "status": "completed",
                    "name": name,
                    "found": True,
                    "result": msg,
                },
            )


def validate(
    project_name: str,
    study_name: str | None = None,
    project_root: str | None = None,
    **_: object,
) -> None:
    """Validate a project/study config set and write a machine-readable JSON result."""
    project_dir = cli_helpers.get_project_dir(project_name, project_root)
    result = _validate_project(project_dir, study_name, validate_configs=True)
    cli_helpers.write_json_result(result)


def inspect_mesh(
    mesh: str,
    machine_config: str | None = None,
    command: str | None = None,
    **_: object,
) -> None:
    """Inspect a mesh file and optionally compare it with a machine.yaml file."""
    from emsopt_analyzer.analyzer.mesh_inspection import inspect_mesh_file

    config = _load_machine_config(Path(machine_config)) if machine_config is not None else None
    result = inspect_mesh_file(mesh, config)
    cli_helpers.emit_command_result(command, {"status": "completed", "result": result})


def validate_mesh(
    project_name: str,
    study_name: str | None = None,
    mesh: str | None = None,
    project_root: str | None = None,
    command: str | None = None,
    **_: object,
) -> None:
    """Run lightweight project/study checks without executing optimization setup."""
    from emsopt_analyzer.analyzer.mesh_inspection import inspect_mesh_file

    project_dir = cli_helpers.get_project_dir(project_name, project_root)
    _validate_project(project_dir)
    config_dir = cli_helpers.get_config_dir(project_name, study_name, project_root)
    machine_config_path = config_dir / BaseConfig.MACHINE_CONFIG_FILENAME.value
    machine_config = _load_machine_config(machine_config_path)
    mesh_path = (
        Path(mesh) if mesh is not None else _resolve_validate_mesh_mesh_path(project_dir, config_dir, machine_config)
    )
    result = inspect_mesh_file(mesh_path, machine_config)
    cli_helpers.emit_command_result(
        command,
        {
            "status": "completed",
            "project": project_name,
            "study": study_name,
            "config_dir": str(config_dir),
            "machine_config": str(machine_config_path),
            "mesh": str(mesh_path),
            "ok": result["machine_checks"]["ok"],
            "result": result,
        },
    )


def _load_machine_config(config_path: Path) -> MachineConfig:
    with config_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        msg = f"machine.yaml root must be a mapping: {config_path}"
        raise TypeError(msg)
    return MachineConfig.model_validate(data)


def _resolve_validate_mesh_mesh_path(project_dir: Path, config_dir: Path, machine_config: MachineConfig) -> Path:
    mesh_filename = machine_config.design_target_filename
    candidates = [
        config_dir / mesh_filename,
        project_dir / mesh_filename,
        config_dir / BaseConfig.RESOURCE_DIRNAME.value / mesh_filename,
        project_dir / BaseConfig.RESOURCE_DIRNAME.value / mesh_filename,
    ]
    optimization_yaml = config_dir / BaseConfig.OPTIMIZATION_CONFIG_FILENAME.value
    if optimization_yaml.exists():
        with optimization_yaml.open(encoding="utf-8") as f:
            optimization_config = yaml.safe_load(f) or {}
        resource_dir = optimization_config.get("resource_dir") if isinstance(optimization_config, dict) else None
        if resource_dir is not None:
            candidates.append(Path(resource_dir) / mesh_filename)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    msg = f"Design target mesh '{mesh_filename}' was not found. Checked: {[str(path) for path in candidates]}"
    raise FileNotFoundError(msg)


def main(argv: Sequence[str] | None = None) -> None:
    """parse agrument and call function"""
    parser = setup_argument_parser()
    args = parser.parse_args(argv)
    try:
        args.func(**vars(args))
    except Exception as exc:
        cli_helpers.emit_command_failure(args.command, exc)
        raise
