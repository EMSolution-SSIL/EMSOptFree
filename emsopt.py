"""
emsopt.py
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

import contextlib
import logging
import shutil
from argparse import ArgumentParser, Namespace
from collections.abc import Sequence
from pathlib import Path

import yaml

from emsopt_engine.configs.base import BaseConfig
from emsopt_engine.setup_project import get_available_config_detail, get_available_config_options, setup_project_files
from manager.optimization_manager import OptimizationManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_argument_parser() -> ArgumentParser:
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    # copy_proj command
    parser_init = subparsers.add_parser("cp_proj", help="This command copies a project.")
    parser_init.add_argument("src_project_name", type=str, help="Name of source project.")
    parser_init.add_argument("dst_project_name", type=str, help="Name of destination new project.")
    parser_init.set_defaults(func=cp_proj)

    # cln_proj command
    parser_init = subparsers.add_parser(
        "cln_proj", help="This command clean up a project by removing intermediate and result dirs."
    )
    parser_init.add_argument("project_name", type=str, help="Name of project.")
    parser_init.set_defaults(func=cln_proj)

    # rm_proj command
    parser_init = subparsers.add_parser("rm_proj", help="This command removes a project.")
    parser_init.add_argument("project_name", type=str, help="Name of project.")
    parser_init.set_defaults(func=rm_proj)

    # save_tpl command
    parser_init = subparsers.add_parser(
        "save_tpl", help="This command saves current optimization_problem config as template."
    )
    parser_init.add_argument("project_name", type=str, help="Name of the project.")
    parser_init.add_argument("tpl_name", type=str, help="Name of template.")
    parser_init.set_defaults(func=save_tpl)

    # load_tpl command
    parser_init = subparsers.add_parser(
        "load_tpl", help="This command loads optimization_problem config from template."
    )
    parser_init.add_argument("project_name", type=str, help="Name of the project.")
    parser_init.add_argument("tpl_name", type=str, help="Name of template to load.")
    parser_init.set_defaults(func=load_tpl)

    # run command
    parser_run = subparsers.add_parser("run", help="This command runs optimization based on project's configurations.")
    parser_run.add_argument("project_name", type=str, help="Name of the project.")
    parser_run.set_defaults(func=run)

    # check command
    parser_run = subparsers.add_parser("check", help="This command checks performed optimization summary.")
    parser_run.add_argument("project_name", type=str, help="Name of the project.")
    parser_run.set_defaults(func=check)

    # show_avl command
    parser_run = subparsers.add_parser(
        "show_avl",
        help="This command shows available options for optimizer, evaluator, level_set_function, and optimization problem function. If -n or --name option is specified, the detail of the available will be displayed",
    )
    parser_run.add_argument("-n", "--name", type=str, default=None, help="Name of the availables to be displayed")
    parser_run.set_defaults(func=show_avl)

    return parser


def cp_proj(args: Namespace) -> None:
    src_dir = Path(BaseConfig.PROJECT_DIR.value) / args.src_project_name
    validate_project(src_dir)
    dst_dir: Path = Path(BaseConfig.PROJECT_DIR.value) / args.dst_project_name
    if dst_dir.exists():
        msg = f"Project folder {dst_dir} already exists."
        raise FileExistsError(msg)
    shutil.copytree(src_dir, dst_dir)
    logger.info("Copied project from %s to %s.", args.src_project_name, args.dst_project_name)


def cln_proj(args: Namespace) -> None:
    # remove default intermediate and result dirs
    for dirname in [
        BaseConfig.RESOURCE_DIRNAME.value,
        BaseConfig.PROGRESS_DIRNAME.value,
        BaseConfig.SUMMARY_DIRNAME.value,
    ]:
        dirpath = Path(BaseConfig.PROJECT_DIR.value) / args.project_name / dirname
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(dirpath)


def rm_proj(args: Namespace) -> None:
    shutil.rmtree(Path(BaseConfig.PROJECT_DIR.value) / args.project_name)


def save_tpl(args: Namespace) -> None:
    template_filepath = Path(BaseConfig.PROJECT_DIR.value) / BaseConfig.PROBLEM_TEMPLATE_FILENAME.value
    project_config_filepath = (
        Path(BaseConfig.PROJECT_DIR.value) / args.project_name / BaseConfig.OPTIMIZATION_PROBLEM_CONFIG_FILENAME.value
    )

    # load current templates
    if template_filepath.exists():
        with template_filepath.open(encoding="utf-8") as f:
            templates: dict | None = yaml.safe_load(f)
        if templates is None:
            templates = {}
    else:
        templates = {}

    # add to template
    with project_config_filepath.open(encoding="utf-8") as f:
        new_template: dict | None = yaml.safe_load(f)
    if args.tpl_name in templates:
        msg = f"Template name {args.tpl_name} already exists."
        raise ValueError(msg)
    templates[args.tpl_name] = new_template

    # save template
    with template_filepath.open("w", encoding="utf-8") as f:
        yaml.safe_dump(templates, f)
    logger.info("Saved template to %s.", str(template_filepath))


def load_tpl(args: Namespace) -> None:
    template_filepath = Path(BaseConfig.PROJECT_DIR.value) / BaseConfig.PROBLEM_TEMPLATE_FILENAME.value
    project_config_filepath = (
        Path(BaseConfig.PROJECT_DIR.value) / args.project_name / BaseConfig.OPTIMIZATION_PROBLEM_CONFIG_FILENAME.value
    )

    # load current templates
    if template_filepath.exists():
        with template_filepath.open(encoding="utf-8") as f:
            templates: dict | None = yaml.safe_load(f)
        if templates is None:
            templates = {}
    else:
        templates = {}

    # find template name and dump
    if args.tpl_name not in templates:
        msg = f"Template name {args.tpl_name} doesn't exists."
        raise ValueError(msg)
    with project_config_filepath.open("w", encoding="utf-8") as f:
        yaml.safe_dump(templates[args.tpl_name], f)
    logger.info("Loaded template %s from %s.", args.tpl_name, str(template_filepath))


def run(args: Namespace) -> None:
    project_dir = Path(BaseConfig.PROJECT_DIR.value) / args.project_name
    validate_project(project_dir)
    # setup injector (optimization setupper)
    injector = setup_project_files(args.project_name)
    # solve optimization-related dependencies via Injector.get
    manager = injector.get(OptimizationManager)
    # run optimization
    manager.run_optimization(
        str(Path(BaseConfig.PROJECT_DIR.value) / args.project_name / BaseConfig.SUMMARY_DIRNAME.value)
    )


def check(args: Namespace) -> None:
    project_dir = Path(BaseConfig.PROJECT_DIR.value) / args.project_name
    validate_project(project_dir)
    # setup injector (optimization setupper)
    injector = setup_project_files(args.project_name)
    # solve optimization-related dependencies via Injector.get
    manager = injector.get(OptimizationManager)
    # check optimization result
    manager.check_optimization_result(
        str(Path(BaseConfig.PROJECT_DIR.value) / args.project_name / BaseConfig.SUMMARY_DIRNAME.value)
    )


def show_avl(args: Namespace) -> None:
    if args.name is None:
        logger.info(get_available_config_options())
    else:
        msg = get_available_config_detail(args.name)
        if msg is None:
            logger.info("No document found for %s.", {args.name})
        else:
            logger.info(msg)


def validate_project(project_dir: Path) -> None:
    """validate existance of project directory"""
    if not project_dir.exists():
        msg = f"Project name {project_dir} doesn't exists."
        raise FileNotFoundError(msg)


def main(argv: Sequence[str] | None = None) -> None:
    """parse agrument and call function"""
    parser = setup_argument_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
