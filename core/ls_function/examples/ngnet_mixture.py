"""
ngnet_mixture.py
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

import numpy as np

from core.ls_function.examples.ngnet import (
    CoordinateSystem,
    Dimension,
    NGnet,
    NGnetValidationError,
    _to_2d_region,
    _to_3d_region,
    _validate_dimension_coordinate,
)
from core.ls_function.examples.simple_level_set_radius import SimpleLevelSetRadius
from emsopt_engine.interface.level_set_function_interface import LevelSetFunctionInterface
from emsopt_engine.registry import level_set_function


def _dimension_name(dimension: int) -> Dimension:
    if dimension == 2:
        return "2D"
    if dimension == 3:
        return "3D"
    msg = f"dimension must be 2 or 3, but got {dimension}."
    raise NGnetValidationError(msg)


class NGnetMixture(LevelSetFunctionInterface):
    def __init__(
        self,
        ngnet: NGnet,
        bound_r: float,
        dimension: int = 2,
        *,
        inversed: bool = False,
        coordinate: CoordinateSystem = "Cartesian",
        fixed_design_region: tuple | list | None = None,
    ) -> None:
        dimension_label = _dimension_name(dimension)
        _validate_dimension_coordinate(dimension_label, coordinate)
        self.ngnet = ngnet
        self.ls_r = SimpleLevelSetRadius(2)
        self.bound_r = bound_r
        self.dimension = dimension
        self.inversed = inversed
        self.coordinate = coordinate
        if fixed_design_region is not None:
            self.ngnet.configure_fixed_design_region(fixed_design_region, dimension_label, coordinate)

    def get_variable_dimension(self) -> int:
        return self.ngnet.get_variable_dimension()

    def calculate_output(self, parameters: np.ndarray, points: np.ndarray) -> np.ndarray:
        if points.shape[1] != self.dimension:
            msg = f"Input dimension {points.shape[1]} does not match expected {self.dimension}."
            raise NGnetValidationError(msg)
        # first, level set by radius
        level_radius = self.ls_r.calculate_output(self.bound_r, points)
        if self.inversed:
            # level set points with r > bound_r, i.e. level_radius < 0
            ngnet_mask = level_radius < 0
        else:
            # level set points with r < bound_r, i.e. level_radius > 0
            ngnet_mask = level_radius > 0
        level = np.ones_like(level_radius)
        if np.any(ngnet_mask):
            level[ngnet_mask] = self.ngnet.calculate_output(parameters, points[ngnet_mask])
        return level


@level_set_function("ngnet_mixture")
def build_ngnet_mixture(
    sigma: float,
    design_region: tuple,
    coordinate: CoordinateSystem,
    boundary_r: float,
    distance_factor: float = 0.8,
    *,
    dimension: Dimension = "2D",
    inversed: bool = False,
    eliminate_bases_on_edge: bool = False,
    normalize_output: bool = True,
    check_basis_GUI: bool = False,
    fixed_design_region: tuple | list | None = None,
) -> LevelSetFunctionInterface:
    """
    NGnet_mixture class. Assumes the covariance matrix has only diagonal components.
    Level is set by the following procedure:
        1: entire region is splitted by boundary_r
        2: for r < bound_r (or r > bound_r when inversed), NGnet is applied

    Args:
        sigma (float): Standard deviation of the Gaussian basis functions to be arranged
        design_region (PhysicalRegion): design region definition.
            2D: ((x1, x2), (y1, y2)) for Cartesian, or ((r1, r2), (theta1, theta2)) for Polar.
            3D: ((x1, x2), (y1, y2), (z1, z2)) for Cartesian,
            or ((r1, r2), (theta1, theta2), (z1, z2)) for Polar.
        coordinate (str): Coordinate system. "Cartesian" or "Polar".
        boundary_r (float): boundary radius to split level set region
        dimension (str): Spatial dimension. "2D" or "3D". Defaults to "2D".
        distance_factor: Value indicating how far apart the centers of the Gaussian basis functions.
            Please refer to NGnet.arrange_gaussian_bases method. Defaults to 0.8.
        inversed (bool): inverse region where NGnet is applied
        eliminate_bases_on_edge (bool, optional): Whether to remove basis functions located on the region boundary.
            Defaults to False.
        normalize_output (bool): If False, normalization of function output will be disabled.
        (so it will be no longer "NGnet", but just weighted sum of Gaussian bases). Defaults to True.
        check_basis_GUI (bool): If True, opens GUI for basis check; otherwise saves image.
        fixed_design_region (PhysicalRegion, optional): Variable region whose outside is fixed to level-set value 1.0.
    """
    ngnet = NGnet(normalize_output=normalize_output, dimension=dimension, coordinate=coordinate)
    _validate_dimension_coordinate(dimension, coordinate)
    ngnet.configure_fixed_design_region(
        fixed_design_region,
        dimension,
        coordinate,
        polar_angles_in_degrees=True,
    )
    # organize argumant
    if dimension == "2D":
        design_region = [list(axis) for axis in _to_2d_region(design_region)]  # type: ignore[arg-type]
    else:
        design_region = [list(axis) for axis in _to_3d_region(design_region)]  # type: ignore[arg-type]
    if coordinate == "Polar":
        design_region[1][0] = np.deg2rad(design_region[1][0])
        design_region[1][1] = np.deg2rad(design_region[1][1])
    # arrange NGnet
    mu = ngnet.arrange_gaussian_bases(
        sigma,
        design_region,
        dimension,
        coordinate,
        distance_factor,
        eliminate_bases_on_edge=eliminate_bases_on_edge,
    )
    ngnet.set_parameters(mu, sigma)
    if dimension == "2D":
        ngnet.visualize_gaussian_bases_radius(
            design_region,
            coordinate,
            "gaussian.png",
            check_basis_GUI=check_basis_GUI,
        )
    else:
        ngnet.visualize_gaussian_bases_radius_3d(
            design_region,
            coordinate,
            "gaussian_3d.png",
            check_basis_GUI=check_basis_GUI,
        )
    return NGnetMixture(
        ngnet,
        boundary_r,
        dimension=3 if dimension == "3D" else 2,
        inversed=inversed,
        coordinate=coordinate,
    )
