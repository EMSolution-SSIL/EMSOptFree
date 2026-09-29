"""
opt_problem_functions.py
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

# --- Examples of objective functions ---
import json
from logging import getLogger
from pathlib import Path

import numpy as np
import pyvista as pv

from emsopt_engine.configs.base import BaseConfig
from emsopt_engine.registry import opt_problem_function

logger = getLogger(__name__)

EPS = 1e-10


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
    json_path = Path(working_dir) / "output.json"
    with json_path.open(encoding="utf-8") as f:
        result = json.load(f)
    torque_wave = np.array(
        [
            item["forceMZ"]
            for item in result["postData"]["forceNodal"]["forceNodalData"]
            if item["propertyNum"] == "rotor"
        ][0]
    )
    torque_wave *= torque_scale
    res = np.mean(torque_wave)
    return res


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
    res = 0 if area < EPS else avg_torque / area
    return res


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
    json_path = Path(working_dir) / "output.json"
    with json_path.open(encoding="utf-8") as f:
        result = json.load(f)
    torque_wave = np.array(
        [
            item["forceMZ"]
            for item in result["postData"]["forceNodal"]["forceNodalData"]
            if item["propertyNum"] == "rotor"
        ][0]
    )
    torque_wave *= torque_scale
    res = np.max(torque_wave) - np.min(torque_wave)
    return res


@opt_problem_function("torque_ripple_percentage")
def torque_ripple_percentage(working_dir: str, torque_scale: float = 1.0) -> float:
    """
    Calculate the torque ripple percentage: (max value - min value) / average × 100 [%].
    Args:
        working_dir (str): Directory containing analysis result json file
        torque_scale (float, optional): Torque scale factor. Defaults to 1.0.
    Returns:
        float: Torque ripple percentage [%]
    """
    json_path = Path(working_dir) / "output.json"
    with json_path.open(encoding="utf-8") as f:
        result = json.load(f)
    torque_wave = np.array(
        [
            item["forceMZ"]
            for item in result["postData"]["forceNodal"]["forceNodalData"]
            if item["propertyNum"] == "rotor"
        ][0]
    )
    torque_wave *= torque_scale
    average_torque = np.mean(torque_wave)
    if abs(average_torque) < 1e-6:
        res = 1e6
    else:
        res = abs(100.0 * (np.max(torque_wave) - np.min(torque_wave)) / average_torque)
    return res


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
    import networkx as nx

    mesh, target_cells = _get_mesh_and_target_cells_gmsh(working_dir, physical_tag)
    if mesh is None or len(target_cells) == 0:
        return 0
    # List of edges for each element
    elem_edges_list = [_get_elem_edges(elem) for elem in target_cells]

    # Dictionary mapping edge to element indices
    edge_to_elems = {}
    for idx, edges in enumerate(elem_edges_list):
        for edge in edges:
            edge_to_elems.setdefault(edge, set()).add(idx)

    # Create adjacency graph (only edge sharing)
    G = nx.Graph()
    for idx in range(len(target_cells)):
        G.add_node(idx)
    for edge, elems in edge_to_elems.items():
        elems = list(elems)
        if len(elems) >= 2:
            # Connect elements that share the same edge
            for i in range(len(elems)):
                for j in range(i + 1, len(elems)):
                    G.add_edge(elems[i], elems[j])

    # Count the number of connected components
    components = list(nx.connected_components(G))
    return len(components)


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

    mesh, target_cells = _get_mesh_and_target_cells_gmsh(working_dir, physical_tag)
    if mesh is None or len(target_cells) == 0:
        return 0.0
    # List of edges for each element
    elem_edges_list = [edge for elem in target_cells for edge in _get_elem_edges(elem)]
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
    res = ds.cell_data["Area"].sum()
    return res


def _get_elem_edges(elem) -> list[tuple[int, int]]:
    """
    Args:
        elem: a list of node indices representing an element (triangle, quad, ...)
    Returns:
        A list of edges represented as tuples, where each tuple contains two node indices.
    """
    n = len(elem)
    return [tuple(sorted((elem[i], elem[(i + 1) % n]))) for i in range(n)]


def _get_mesh_and_target_cells_gmsh(working_dir: str, physical_tag: int) -> tuple:
    """
    Get target cells based on physical tag for both triangle and quad elements.
    Args:
        working_dir (str): Directory containing Mesh file (.msh)
        physical_tag (int): Target physical tag number
    Returns:
        mesh (meshio.Mesh): The mesh object containing the target cells.
        np.ndarray: Array of all target cells (triangle, quad)
    """
    import meshio

    mesh = meshio.read(str(Path(working_dir) / BaseConfig.SAVE_MESH_FILENAME.value))
    target_cells = []
    for etype in ("triangle", "quad"):
        if etype in mesh.cells_dict and etype in mesh.cell_data_dict["gmsh:physical"]:
            cells = mesh.cells_dict[etype]
            phys = mesh.cell_data_dict["gmsh:physical"][etype]
            target_idx = np.where(phys == physical_tag)[0]
            target_cells.extend(cells[target_idx])
    if len(target_cells) == 0:
        logger.warning("No triangle or quad elements found with physical tag %d.", physical_tag)
        return None, []
    return mesh, target_cells
