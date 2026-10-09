"""
opt_problem_functions.py
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

# --- Examples of objective functions ---
import json
from logging import getLogger
from pathlib import Path

import meshio
import numpy as np
import pyvista as pv

from emsopt_engine.configs.base import BaseConfig
from emsopt_engine.registry import opt_problem_function

logger = getLogger(__name__)

INV_VALUE = 1e6
INV_VALUE_STRESS = 1e9
EPS = 1e-10
EPS_T = 1e-6


def _obtain_torque_wave(working_dir: str, torque_scale: float = 1.0) -> np.ndarray:
    json_path = Path(working_dir) / "output.json"
    with json_path.open(encoding="utf-8") as f:
        result = json.load(f)
    torque_wave = np.array(
        next(
            item["forceMZ"]
            for item in result["postData"]["forceNodal"]["forceNodalData"]
            if item["propertyNum"] == "rotor"
        )
    )
    torque_wave *= torque_scale
    return torque_wave


@opt_problem_function("average_torque")
def average_torque(working_dir: str, torque_scale: float = 1.0) -> float:
    """
    Calculate the average value of the torque waveform.
    Args:
        working_dir (str): Directory containing analysis result json file
        torque_scale (float, optional): Torque scale factor. Defaults to 1.0.
    Returns:
        float: Average torque value
    """
    torque_wave = _obtain_torque_wave(working_dir, torque_scale)
    return np.mean(torque_wave)


@opt_problem_function("torque_density")
def torque_density(working_dir: str, physical_tag: int = 20) -> float:
    """
    Calculate torque density, that is {average torque} / {total area of designated material}
    Args:
        working_dir (str): Directory containing analysis result json file
        physical_tag (int): Target physical tag number

    Returns:
        float: Torque density
    """
    avg_torque = average_torque(working_dir)
    area = material_area(working_dir, physical_tag)
    return 0 if area < EPS else avg_torque / area


@opt_problem_function("torque_ripple")
def torque_ripple(working_dir: str, torque_scale: float = 1.0) -> float:
    """
    Calculate the torque ripple (max value - min value) of the torque waveform.
    Args:
        working_dir (str): Directory containing analysis result json file
        torque_scale (float, optional): Torque scale factor. Defaults to 1.0.
    Returns:
        float: Torque ripple value
    """
    torque_wave = _obtain_torque_wave(working_dir, torque_scale)
    return np.max(torque_wave) - np.min(torque_wave)


@opt_problem_function("torque_ripple_percentage")
def torque_ripple_percentage(working_dir: str, torque_scale: float = 1.0) -> float:
    """
    Calculate the torque ripple percentage: (max value - min value) / average x 100 [%].
    Args:
        working_dir (str): Directory containing analysis result json file
        torque_scale (float, optional): Torque scale factor. Defaults to 1.0.
    Returns:
        float: Torque ripple percentage [%]
    """
    torque_wave = _obtain_torque_wave(working_dir, torque_scale)
    average_torque = np.mean(torque_wave)
    if abs(average_torque) < EPS_T:
        res = INV_VALUE
    else:
        res = abs(100.0 * (np.max(torque_wave) - np.min(torque_wave)) / average_torque)
    return res


@opt_problem_function("torque_squared_error")
def torque_squared_error(working_dir: str, torque_target: float, torque_scale: float = 1.0) -> float:
    """
    Calculate the torque squared_error from target value.
    Args:
        working_dir (str): Directory containing analysis result json file
        torque_target (float): Torque target value.
        torque_scale (float, optional): Torque scale factor. Defaults to 1.0.
    Returns:
        float: Torque ripple value
    """
    torque_wave = _obtain_torque_wave(working_dir, torque_scale)
    return np.sum((torque_wave - torque_target) ** 2)


@opt_problem_function("maximum_voltage")
def maximum_voltage(working_dir: str) -> float:
    """
    Calculate the maximum voltage.
    Args:
        working_dir (str): Directory containing analysis result json file
    Returns:
        float: maximum voltage [V]
    """
    json_path = Path(working_dir) / "output.json"
    with json_path.open(encoding="utf-8") as f:
        result = json.load(f)
    voltage_waves = [
        item["voltage"][1:] for item in result["postData"]["network"]["networkData"] if item["elementName"] == "FEM"
    ]  # omit first value for stable evaluation
    return np.abs(voltage_waves).max()


@opt_problem_function("magnetic_energy")
def magnetic_energy(working_dir: str, physical_tag: int) -> float:
    """
    Read magnetic energy of designated material
    Args:
        working_dir (str): Directory containing analysis result mesh file
        physical_tag (int): Target physical tag number
    Returns:
        float: magnetic energy of designated material
    """
    json_path = Path(working_dir) / "output.json"
    with json_path.open(encoding="utf-8") as f:
        result = json.load(f)
    try:
        energy = next(
            item["energy"]
            for item in result["postData"]["magneticEnergy"]["magneticEnergyData"]
            if int(item["propertyNum"]) == physical_tag
        )
    except KeyError as e:
        msg = f"""
            Magnetic energy for material {physical_tag} does not exist in post data.
            Please check input json file for pyemsol.
            """
        raise KeyError(msg) from e
    energy_values = np.asarray(energy, dtype=float)
    if energy_values.size == 0:
        msg = f"Magnetic energy for material {physical_tag} is empty."
        raise ValueError(msg)
    return float(np.mean(energy_values))


@opt_problem_function("num_connected_components")
def num_connected_components(working_dir: str, physical_tag: int = 20) -> int:
    """
    Count the number of connected components by edge sharing.

    Args:
        working_dir (str): Directory containing Mesh file (.msh)
        physical_tag (int): Target physical tag number

    Returns:
        int: Number of connected components
    """
    mesh = meshio.read(Path(working_dir) / BaseConfig.SAVE_MESH_FILENAME.value)
    target_cells = get_target_cells(mesh, physical_tag)
    if mesh is None or len(target_cells) == 0:
        return 0
    return get_num_connected_components(mesh, target_cells)


@opt_problem_function("boundary_length")
def boundary_length(working_dir: str, physical_tag: int = 20) -> float:
    """

    Args:
        working_dir (str): Directory containing Mesh file (.msh)
        physical_tag (int): Target physical tag number
    Returns:
        float: boundary length
    """
    from collections import Counter

    mesh = meshio.read(Path(working_dir) / BaseConfig.SAVE_MESH_FILENAME.value)
    target_cells = get_target_cells(mesh, physical_tag)
    if mesh is None or len(target_cells) == 0:
        return 0.0
    # List of edges for each element
    elem_edges_list = [edge for elem in target_cells for edge in get_elem_edges(elem)]
    edge_count = Counter(elem_edges_list)
    # extract boundary edges (those that appear only once)
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]

    total_length = 0.0
    for edge in boundary_edges:
        p1, p2 = mesh.points[list(edge)]
        total_length += np.linalg.norm(p1 - p2)

    return total_length


@opt_problem_function("material_area")
def material_area(working_dir: str, physical_tag: int = 20) -> float:
    """
    Calculate total area of designated material
    Args:
        working_dir (str): Directory containing analysis result mesh file
        physical_tag (int): Target physical tag number
    Returns:
        float: area of designated material
    """
    mesh = pv.read(str(Path(working_dir) / BaseConfig.SAVE_MESH_FILENAME.value))
    material_ids = mesh.cell_data["gmsh:physical"]
    target_cells = mesh.extract_cells(material_ids == physical_tag)
    surf = target_cells.extract_surface()
    ds = surf.compute_cell_sizes(area=True, volume=False)
    return ds.cell_data["Area"].sum()


@opt_problem_function("material_volume")
def material_volume(working_dir: str, physical_tag: int = 20) -> float:
    """
    Calculate total volume of designated material
    Args:
        working_dir (str): Directory containing analysis result mesh file
        physical_tag (int): Target physical tag number
    Returns:
        float: volume of designated material
    """
    mesh = pv.read(str(Path(working_dir) / BaseConfig.SAVE_MESH_FILENAME.value))
    material_ids = mesh.cell_data["gmsh:physical"]
    target_cells = mesh.extract_cells(material_ids == physical_tag)
    ds = target_cells.compute_cell_sizes(area=False, volume=True)
    return ds.cell_data["Volume"].sum()


@opt_problem_function("magnet_cost")
def magnet_cost(working_dir: str, physical_tag: int = 50000, ferrite_coef: float = 0.2) -> float:
    """
    Calculate pseudo cost for magnet (= coefficient * area)
    Args:
        working_dir (str): Directory containing analysis result mesh file
        physical_tag (int): Magnet physical tag number
        ferrite_coef (float): coefficient for low-cost ferrite magnet
    Returns:
        float: pseudo cost for magnet
    """
    json_path = Path(working_dir).parent / "ems_project.json"  # seek for eMotorSolution project file
    with json_path.open(encoding="utf-8") as f:
        input_json = json.load(f)
    magnet_material = input_json["rotor"]["hole_magnet"]["collection"][0]["_magnet_material"]
    if magnet_material == "NdFeB":
        coef = 1.0
    elif magnet_material == "FerriteMagnet":
        coef = ferrite_coef
    else:
        logger.warning("No assumed magnet data found.")
        return INV_VALUE
    area = material_area(working_dir, physical_tag)
    return coef * area


@opt_problem_function("von_mises_stress")
def von_mises_stress(working_dir: str, *, prohibit_seperated: bool = False) -> float:
    """
    Read von Mises stress scalar values from structural analysis result mesh.

    Args:
        working_dir (str): Directory containing structural_displacement.vtk
        prohibit_seperated (bool): If True, prohibit seperated mesh.
    Returns:
        float: Maximum von Mises stress value
    """
    mesh_path = Path(working_dir) / "structural_displacement.vtk"
    mesh = meshio.read(mesh_path)
    if prohibit_seperated:
        target_cells = get_target_cells(mesh, None)
        n = get_num_connected_components(mesh, target_cells)
        if n > 1:
            return INV_VALUE_STRESS

    stress_values = []
    if "von_mises" in mesh.point_data:
        stress_values.append(np.asarray(mesh.point_data["von_mises"], dtype=float).ravel())
    if "von_mises" in mesh.cell_data:
        stress_values.extend(np.asarray(values, dtype=float).ravel() for values in mesh.cell_data["von_mises"])
    if not stress_values:
        msg = f"Scalar data 'von_mises' does not exist in mesh: {mesh_path}"
        raise KeyError(msg)

    non_empty_values = [values for values in stress_values if values.size > 0]
    if not non_empty_values:
        msg = f"Scalar data 'von_mises' is empty in mesh: {mesh_path}"
        raise ValueError(msg)
    values = np.concatenate(non_empty_values)
    return float(np.max(values))


def get_num_connected_components(mesh: meshio.Mesh, target_cells: np.ndarray) -> int:
    import networkx as nx

    # List of edges for each element
    elem_edges_list = [get_elem_edges(elem) for elem in target_cells]

    # Dictionary mapping edge to element indices
    edge_to_elems = {}
    for idx, edges in enumerate(elem_edges_list):
        for edge in edges:
            edge_to_elems.setdefault(edge, set()).add(idx)

    # Create adjacency graph (only edge sharing)
    graph = nx.Graph()
    for idx in range(len(target_cells)):
        graph.add_node(idx)
    for elems in edge_to_elems.values():
        elems_list = list(elems)
        if len(elems_list) >= 2:  # noqa: PLR2004
            # Connect elements that share the same edge
            for i in range(len(elems_list)):
                for j in range(i + 1, len(elems_list)):
                    graph.add_edge(elems_list[i], elems_list[j])

    # Count the number of connected components
    components = list(nx.connected_components(graph))
    return len(components)


def get_elem_edges(elem: list[int]) -> list[tuple[int, int]]:
    """
    Args:
        elem: a list of node indices representing an element (triangle, quad, ...)
    Returns:
        A list of edges represented as tuples, where each tuple contains two node indices.
    """
    n = len(elem)
    return [tuple(sorted((elem[i], elem[(i + 1) % n]))) for i in range(n)]


def get_target_cells(mesh: meshio.Mesh, physical_tag: int | None = None) -> np.ndarray:
    """
    Get target cells based on physical tag for both triangle and quad elements.
    Args:
        mesh (meshio.Mesh): target mesh
        physical_tag (int | None): Target physical tag number. If None, get all cells.
            Can be treated when mesh has gmsh format.
    Returns:
        np.ndarray: Array of all target cells (triangle, quad)
    """
    target_cells = []
    for etype in ("triangle", "quad"):
        if etype in mesh.cells_dict:
            cells = mesh.cells_dict[etype]
            if physical_tag is not None:
                phys = mesh.cell_data_dict["gmsh:physical"][etype]
                target_idx = np.where(phys == physical_tag)[0]
                target_cells.extend(cells[target_idx])
            else:
                target_cells.extend(cells)
    if len(target_cells) == 0:
        logger.warning("No triangle or quad elements found with physical tag %s.", physical_tag)
        return []
    return target_cells
