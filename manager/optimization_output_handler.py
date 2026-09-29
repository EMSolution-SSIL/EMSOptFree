"""
optimization_output_handler.py
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
import shutil
from abc import ABC, abstractmethod
from dataclasses import asdict
from logging import getLogger
from pathlib import Path

import meshio
import numpy as np
import pandas as pd
import pyvista as pv
import yaml

from emsopt_engine.configs.optimization import OutputConfig
from emsopt_engine.individual import Individual, OptimizationProblemMetrics, Population
from emsopt_engine.interface.mo_optimizer_base import MOOptimizerBase
from emsopt_engine.interface.optimizer_interface import OptimizerInterface
from emsopt_engine.interface.so_optimizer_base import SOOptimizerBase
from manager.optimization_analyzer import OptimizationAnalyzer

logger = getLogger(__name__)

# add yaml representer
yaml.add_multi_representer(
    np.floating, lambda dumper, value: dumper.represent_float(float(value)), Dumper=yaml.SafeDumper
)
yaml.add_multi_representer(Path, lambda dumper, value: dumper.represent_str(str(value)), Dumper=yaml.SafeDumper)


class OptimizationOutputHandlerInterface(ABC):
    """
    Interface for optimization handler.

    This abstract base class defines the required methods for outputting optimization results,
    such as best individuals, outcomes, candidate plots, and creating visualization payloads.
    Implementations should provide concrete logic for single-objective or multi-objective optimization workflows.
    """

    @abstractmethod
    def output_best_individual(
        self, optimizer: OptimizerInterface, output_dir: Path, i_iter: int, outconf: OutputConfig
    ) -> None:
        """
        Output the best individuals of the current optimization state.

        Args:
            optimizer (OptimizerInterface): The optimizer instance.
            output_dir (Path): Directory to save output files.
            i_iter (int): Current iteration number.
            outconf (OutputConfig): Output configuration.
        """
        raise NotImplementedError

    @abstractmethod
    def output_candidate(self, candidates: Population, output_dir: Path, i_iter: int, outconf: OutputConfig) -> None:
        """
        Output candidate information

        Args:
            candidates (Population): Candidate solutions.
            output_dir (Path): Directory to save output files.
            i_iter (int): Current iteration number.
            outconf (OutputConfig): Output configuration.
        """
        raise NotImplementedError

    @abstractmethod
    def output_candidate_plot(
        self, candidates: Population, output_dir: Path, i_iter: int, outconf: OutputConfig
    ) -> None:
        """
        Output a plot of candidate solutions for the current iteration.

        Args:
            candidates (Population): Candidate solutions.
            output_dir (Path): Directory to save output files.
            i_iter (int): Current iteration number.
            outconf (OutputConfig): Output configuration.
        """
        raise NotImplementedError

    @abstractmethod
    def save_latest_summary_dir(self, dst_dir: str) -> None:
        """save latest summary dir to dst_dir

        Args:
            dst_dir (str): destination directory
        """
        raise NotImplementedError


class SOHandler(OptimizationOutputHandlerInterface):
    """
    Single-objective specific output handler.

    Handles output operations for single-objective optimization, including saving best individuals,
    outcomes, candidate plots, and creating visualization payloads for the optimization process.
    """

    def __init__(self) -> None:
        """
        Initialize SOHandler.
        """
        self.analyzer = OptimizationAnalyzer()
        self.latest_result_dir: str = ""
        self._candidate_csv_files: list[str] = []

    def output_best_individual(
        self, optimizer: SOOptimizerBase, output_dir: Path, i_iter: int, outconf: OutputConfig
    ) -> None:
        """
        Output the best individual to CSV files.

        Args:
            optimizer (SOOptimizerBase): The optimizer instance.
            output_dir (Path): Directory to save output files.
            i_iter (int): Current iteration number.
            outconf (OutputConfig): Output configuration.
        """
        if optimizer.best_history:
            # output outcome first
            iter_label = f"iter{i_iter + 1}_"
            if optimizer.best_history:
                out_f = str(output_dir / f"{iter_label}{outconf.filename_base}")
                # to copy in next output, set the filepath
                optimizer.best_history[-1].outcome_filepath = output_outcome(optimizer.best_history[-1], out_f)
            # dump individual's info
            yaml_f = output_dir / outconf.filename_base
            dump_individuals(optimizer.best_history, str(yaml_f.with_suffix(".yaml")))
            dump_individuals_csv(optimizer.best_history, str(yaml_f.with_suffix(".csv")))
        self.latest_result_dir = str(output_dir)

    def output_candidate(self, candidates: Population, output_dir: Path, i_iter: int, outconf: OutputConfig) -> None:
        """
        Output candidate information

        Args:
            candidates (Population): Candidate solutions.
            output_dir (Path): Directory to save output files.
            i_iter (int): Current iteration number.
            outconf (OutputConfig): Output configuration.
        """
        if len(candidates) <= 0:
            return
        iter_label = f"iter{i_iter + 1}_"
        for i in range(len(candidates)):
            out_f = str(output_dir / f"{iter_label}{i + 1}_{outconf.filename_base}")
            _ = output_outcome(candidates[i], out_f)
        yaml_f = output_dir / f"{iter_label}{outconf.filename_base}"
        dump_individuals(list(candidates.values()), str(yaml_f.with_suffix(".yaml")))
        csv_path = str(yaml_f.with_suffix(".csv"))
        dump_individuals_csv(list(candidates.values()), csv_path)
        self._candidate_csv_files.append(csv_path)

    def output_candidate_plot(
        self, candidates: Population, output_dir: Path, i_iter: int, outconf: OutputConfig
    ) -> None:
        """
        Output a plot of candidate solutions for the current iteration.

        Args:
            candidates (Population): Candidate solutions.
            output_dir (Path): Directory to save output files.
            i_iter (int): Current iteration number.
            outconf (OutputConfig): Output configuration.
        """
        if len(candidates) <= 0:
            return
        if self.analyzer.pca is None:
            self.analyzer.fit_pca(candidates)
        plot_f = output_dir / f"iter{i_iter + 1}_{outconf.filename_base}"
        self.analyzer.plot_population(candidates, str(plot_f.with_suffix(".png")))

    def save_latest_summary_dir(self, dst_dir: str) -> None:
        """save latest summary dir to dst_dir

        Args:
            dst_dir (str): destination directory
        """
        if self.latest_result_dir != "":
            shutil.copytree(self.latest_result_dir, dst_dir)
        merge_candidate_csv_files(self._candidate_csv_files, dst_dir)


class MOHandler(OptimizationOutputHandlerInterface):
    """
    Multi-objective specific output handler.

    Handles output operations for multi-objective optimization, such as saving Pareto front individuals,
    outcomes, candidate plots, and creating visualization payloads for the optimization process.
    """

    def __init__(self) -> None:
        """
        Initialize MOHandler.
        """
        self.analyzer = OptimizationAnalyzer()
        self.latest_result_dir: str = ""
        self._candidate_csv_files: list[str] = []

    def output_best_individual(
        self, optimizer: MOOptimizerBase, output_dir: Path, i_iter: int, outconf: OutputConfig
    ) -> None:
        """
        Output the Pareto front individuals to CSV files for the current iteration.

        Args:
            optimizer (MOOptimizerBase): The optimizer instance.
            output_dir (Path): Directory to save output files.
            i_iter (int): Current iteration number.
            outconf (OutputConfig): Output configuration.
        """
        if optimizer.pareto:
            output_subdir = output_dir / f"iter{i_iter + 1}"
            output_subdir.mkdir(exist_ok=True)
            # output outcome first
            for i, ind in enumerate(optimizer.pareto.values()):
                out_f = str(output_subdir / f"pareto{i + 1}_{outconf.filename_base}")
                # to copy in next output, set the filepath
                ind.outcome_filepath = output_outcome(ind, out_f)
            # dump individual's info
            yaml_f = output_subdir / outconf.filename_base
            dump_individuals(list(optimizer.pareto.values()), str(yaml_f.with_suffix(".yaml")))
            dump_individuals_csv(list(optimizer.pareto.values()), str(yaml_f.with_suffix(".csv")))
            self.latest_result_dir = str(output_subdir)

    def output_candidate(self, candidates: Population, output_dir: Path, i_iter: int, outconf: OutputConfig) -> None:
        """
        Output candidate information

        Args:
            candidates (Population): Candidate solutions.
            output_dir (Path): Directory to save output files.
            i_iter (int): Current iteration number.
            outconf (OutputConfig): Output configuration.
        """
        if len(candidates) <= 0:
            return
        iter_label = f"iter{i_iter + 1}_"
        yaml_f = output_dir / f"{iter_label}{outconf.filename_base}"
        dump_individuals(list(candidates.values()), str(yaml_f.with_suffix(".yaml")))
        csv_path = str(yaml_f.with_suffix(".csv"))
        dump_individuals_csv(list(candidates.values()), csv_path)
        self._candidate_csv_files.append(csv_path)

    def output_candidate_plot(
        self, candidates: Population, output_dir: Path, i_iter: int, outconf: OutputConfig
    ) -> None:
        """
        Output a plot of candidate solutions for the current iteration.

        Args:
            candidates (Population): Candidate solutions.
            output_dir (Path): Directory to save output files.
            i_iter (int): Current iteration number.
            outconf (OutputConfig): Output configuration.
        """
        if len(candidates) <= 0:
            return
        if self.analyzer.pca is None:
            self.analyzer.fit_pca(candidates)
        plot_f = output_dir / f"iter{i_iter + 1}_{outconf.filename_base}"
        self.analyzer.plot_population(candidates, str(plot_f.with_suffix(".png")))

    def save_latest_summary_dir(self, dst_dir: str) -> None:
        """save latest summary dir to dst_dir

        Args:
            dst_dir (str): destination directory
        """
        if self.latest_result_dir != "":
            shutil.copytree(self.latest_result_dir, dst_dir)
        merge_candidate_csv_files(self._candidate_csv_files, dst_dir)


def merge_candidate_csv_files(csv_files: list[str], dst_dir: str) -> None:
    """
    指定されたCSVファイル群を統合し、dst_dir/merged_candidates.csvに保存する
    Args:
        csv_files (list[str]): CSVファイルパスリスト
        dst_dir (str): 保存先ディレクトリ
    """
    import pandas as pd

    dfs = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            dfs.append(df)
        except Exception as e:
            logger.warning("Failed to read %s: %s", csv_path, e)
    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        dst_csv = Path(dst_dir) / "merged_candidates.csv"
        merged_df.to_csv(dst_csv, index=False)


def dump_individuals(individuals: list[Individual], filepath: str) -> None:
    """Dump individuals to yaml file using yaml.safe_dump

    Args:
        individuals (list[Individual]): individuals to dump
        filepath (str): dst filepath
    """
    dicts = [asdict(ind) for ind in individuals]
    with Path(filepath).open(mode="w", encoding="utf-8") as f:
        yaml.safe_dump(dicts, f)


def dump_individuals_csv(individuals: list[Individual], filepath: str) -> None:
    """Dump individuals to csv file using yaml.safe_dump

    Args:
        individuals (list[Individual]): individuals to dump
        filepath (str): dst filepath
    """
    # convert to dataframe-like list
    records = []
    for ind in individuals:
        d_expanded = {
            "fitness": ind.metrics.fitness,
            "constraint_violation": ind.metrics.constraint_violation,
            **{f"objective_{i + 1}": v for i, v in enumerate(ind.metrics.objectives)},
            **{f"ineq_constraint_{i + 1}": v for i, v in enumerate(ind.metrics.ineq_constraints)},
            **{f"eq_contraint_{i + 1}": v for i, v in enumerate(ind.metrics.eq_constraints)},
            **{f"other_metrics_{i + 1}": v for i, v in enumerate(ind.metrics.other_metrics)},
            **{f"solution_{i + 1}": v for i, v in enumerate(ind.solution)},
        }
        records.append(d_expanded)
    # create pandas DataFrame and convert it to csv
    df = pd.DataFrame(records)
    df.to_csv(filepath)


def load_individuals(filepath: str) -> list[Individual]:
    """Load individuals from yaml file using yaml.safe_load

    Args:
        filepath (str): src filepath

    Returns:
        list[Individual]: individuals loaded
    """
    with Path(filepath).open(mode="r", encoding="utf-8") as f:
        dicts = yaml.safe_load(f)
    individuals = []
    for d in dicts:
        metrics = OptimizationProblemMetrics(**d["metrics"])
        individual = Individual(solution=d["solution"], outcome_filepath=d["outcome_filepath"], metrics=metrics)
        individuals.append(individual)
    return individuals


def output_outcome(individual: Individual, filepath: str) -> str | None:
    """output outcome of individual

    Args:
        filepath (str): filepath for outcome data.

    Returns:
        str: filepath of copied outcome
    """
    if individual.outcome_filepath is not None:
        individual_filepath = Path(individual.outcome_filepath)
        _filepath = Path(filepath)
        dst_filepath = str(_filepath.with_suffix(individual_filepath.suffix))
        with contextlib.suppress(shutil.SameFileError):  # trial of copying same file is allowed
            shutil.copy(individual_filepath, dst_filepath)
        # if data is msh, convert it to png image
        if individual_filepath.suffix == ".msh":
            convert_msh_to_img(
                str(_filepath.with_suffix(".msh")),
                str(Path(dst_filepath).with_suffix(".png")),
                cmap="MotorLikeDiscrete",
            )
    else:
        dst_filepath = None
    return dst_filepath


def convert_msh_to_img(msh_filepath: str, out_filepath: str, cmap: str = "plasma") -> None:
    """visualize mesh data in msh format, and output as image file

    Args:
        msh_filepath (str): msh file path
        out_filepath (str): output image file path
        cmap (str): pyvista color map. Special keyword "MotorLikeDiscrete" maps material numbers to several color groups
    """

    def _convert_msh_to_vtk_celldata(mesh: meshio.Mesh):
        # required data
        all_cells = []
        all_celltypes = []
        materials = []
        vtk_type_map = {
            "triangle": 5,  # VTK_TRIANGLE
            "quad": 9,  # VTK_QUAD
        }
        # convert msh -> vtk
        for i, cell_block in enumerate(mesh.cells):
            ctype = cell_block.type
            if ctype not in ["triangle", "quad"]:
                continue
            c_data = cell_block.data
            n = c_data.shape[1]
            cell_with_size = np.hstack([np.full((c_data.shape[0], 1), n), c_data])
            all_cells.append(cell_with_size.flatten())
            all_celltypes.append(np.full(c_data.shape[0], vtk_type_map[ctype]))
            if "gmsh:physical" in mesh.cell_data:
                mat_id = mesh.cell_data["gmsh:physical"][i]
            else:
                logger.warning("No material info")
            materials.append(mat_id)
        cells_vtk = np.concatenate(all_cells).astype(np.int64)
        celltypes_vtk = np.concatenate(all_celltypes)
        materials_vtk = np.concatenate(materials)
        return cells_vtk, celltypes_vtk, materials_vtk

    def _generate_new_cmap(scalars: np.ndarray):
        cats = np.zeros_like(scalars, dtype=int)
        colors = ["white", "red", "gray"]
        cats[scalars >= 600000] = 0
        cats[scalars < 600000] = 0
        cats[scalars < 100000] = 1
        cats[scalars < 50000] = 0
        cats[scalars <= 20] = 2
        cats[scalars < 10] = 0
        return cats, colors

    # convert msh for pyvista
    mesh = meshio.read(msh_filepath)
    points = mesh.points
    cells_vtk, celltypes_vtk, materials_vtk = _convert_msh_to_vtk_celldata(mesh)
    ugrid = pv.UnstructuredGrid(cells_vtk, celltypes_vtk, points)
    ugrid.cell_data["MaterialID"] = materials_vtk
    # create img
    plotter = pv.Plotter(off_screen=True)
    if cmap == "MotorLikeDiscrete":
        cats, colors = _generate_new_cmap(scalars=np.array(materials_vtk))
        ugrid.cell_data["cats"] = cats
        plotter.add_mesh(
            ugrid, scalars="cats", cmap=colors, clim=(0, len(colors) - 1), show_edges=False, show_scalar_bar=False
        )
    else:
        plotter.add_mesh(ugrid, scalars="MaterialID", cmap=cmap, show_edges=False, show_scalar_bar=False)
    plotter.view_xy()
    plotter.screenshot(out_filepath, window_size=[960, 960])
    plotter.close()
    plotter.clear()
