"""
ngnet_multi_material.py
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

from copy import deepcopy

import numpy as np

from core.ls_function.examples.ngnet import (
    CoordinateSystem,
    Dimension,
    NGnet,
    _to_2d_region,
    _to_3d_region,
    _validate_dimension_coordinate,
)
from emsopt_engine.interface.level_set_function_interface import LevelSetFunctionInterface
from emsopt_engine.registry import level_set_function


class NGnetMultiMaterial(NGnet):
    def __init__(
        self,
        angle_1: float = 120.0,
        angle_2: float = 120.0,
        *,
        normalize_output: bool = True,
        insert_circle: bool = False,
        circle_radius: float = 0.5,
        insert_third_ngnet: bool = False,
        dimension: Dimension = "2D",
        coordinate: CoordinateSystem = "Cartesian",
        fixed_design_region: tuple | list | None = None,
    ) -> None:
        super().__init__(
            normalize_output=normalize_output,
            dimension=dimension,
            coordinate=coordinate,
            fixed_design_region=fixed_design_region,
        )
        self.angle_1 = angle_1
        self.angle_2 = angle_2
        self.insert_circle = insert_circle
        self.circle_radius = circle_radius
        self.insert_third_ngnet = insert_third_ngnet

    def get_variable_dimension(self) -> int:
        return self.K * 3 if self.insert_third_ngnet else self.K * 2

    def calculate_output(self, parameters: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Batch compute the output of NGnet
        Args:
            parameters (np.ndarray): Weight coefficients
            points (np.ndarray): Matrix of calculation points
        Returns:
            np.ndarray: Output calculation result
        """

        def calculate(variable_points: np.ndarray) -> np.ndarray:
            normalized_gaussian = self.calculate_ngnet(variable_points).T
            ngnet_1 = np.dot(parameters[: self.K], normalized_gaussian)
            ngnet_2 = np.dot(parameters[self.K : 2 * self.K], normalized_gaussian)
            if self.insert_third_ngnet:
                ngnet_3 = np.dot(parameters[2 * self.K : 3 * self.K], normalized_gaussian)
            radii = np.sqrt(ngnet_1 * ngnet_1 + ngnet_2 * ngnet_2)
            angles = np.degrees(np.arctan2(ngnet_2, ngnet_1))
            levels = np.zeros_like(angles)
            # starting from -180.0 deg, assign three levels
            if self.insert_third_ngnet:
                if self.insert_circle:
                    levels[angles < -180.0 + self.angle_1] = 0
                    levels[angles >= -180.0 + self.angle_1] = -0.4
                    levels[angles >= -180.0 + self.angle_1 + self.angle_2] = -0.8
                    levels[radii < self.circle_radius] = 0.4
                    levels[ngnet_3 > 0] = 0.8
                else:
                    levels[angles < -180.0 + self.angle_1] = 0.25
                    levels[angles >= -180.0 + self.angle_1] = -0.25
                    levels[angles >= -180.0 + self.angle_1 + self.angle_2] = -0.75
                    levels[ngnet_3 > 0] = 0.75
            else:
                if self.insert_circle:
                    levels[angles < -180.0 + self.angle_1] = 0.25
                    levels[angles >= -180.0 + self.angle_1] = -0.25
                    levels[angles >= -180.0 + self.angle_1 + self.angle_2] = -0.75
                    levels[radii < self.circle_radius] = 0.75
                else:
                    levels[angles < -180.0 + self.angle_1] = 1
                    levels[angles >= -180.0 + self.angle_1] = 0
                    levels[angles >= -180.0 + self.angle_1 + self.angle_2] = -1
            return levels

        return self._calculate_output_in_fixed_design_region(points, calculate)


@level_set_function("ngnet_multi_material")
def build_ngnet_multi_material(
    coordinate: CoordinateSystem,
    sigma: float,
    design_region: tuple,
    distance_factor: float = 0.8,
    angle_1: float = 120.0,
    angle_2: float = 120.0,
    *,
    dimension: Dimension = "2D",
    eliminate_bases_on_edge: bool = False,
    normalize_output: bool = True,
    insert_circle: bool = False,
    circle_radius: float = 0.5,
    insert_third_ngnet: bool = False,
    check_basis_GUI: bool = False,
    fixed_design_region: tuple | list | None = None,
) -> LevelSetFunctionInterface:
    """
    NGnet class. Assumes the covariance matrix has only diagonal components.

    Args:
        coordinate (str): Coordinate system. "Cartesian" or "Polar".
        sigma (float): Standard deviation of the Gaussian basis functions to be arranged
        design_region (PhysicalRegion): design region definition.
            2D: [[x1, x2], [y1, y2]] for Cartesian, or [[r1, r2], [theta1, theta2]] for Polar.
            3D: [[x1, x2], [y1, y2], [z1, z2]] for Cartesian,
            or [[r1, r2], [theta1, theta2], [z1, z2]] for Polar.
        dimension (str): Spatial dimension. "2D" or "3D". Defaults to "2D".
        distance_factor: Value indicating how far apart the centers of the Gaussian basis functions.
            Please refer to NGnet.arrange_gaussian_bases method. Defaults to 0.8.
        angle_1 (float): angle for first material on multimaterial map.
        angle_2 (float): angle for second material on multimaterial map.
        eliminate_bases_on_edge (bool, optional): Whether to remove basis functions located on the region boundary.
            Defaults to False.
        normalize_output (bool): If False, normalization of function output will be disabled.
            (so it will be no longer "NGnet", but just weighted sum of Gaussian bases). Defaults to True.
        insert_circle (bool): insert circle into the representation space.
            In this case, the number of represented material is 4.
        circle_radius (float): circle radius when insert_circle is True.
        insert_third_ngnet (bool): insert third ngnet to represent another material state
        check_basis_GUI (bool): If True, opens GUI for basis check; otherwise saves image.
        fixed_design_region (PhysicalRegion, optional): Variable region whose outside is fixed to level-set value 1.0.
    """
    ngnet = NGnetMultiMaterial(
        angle_1,
        angle_2,
        normalize_output=normalize_output,
        insert_circle=insert_circle,
        circle_radius=circle_radius,
        insert_third_ngnet=insert_third_ngnet,
        dimension=dimension,
        coordinate=coordinate,
    )
    ngnet.configure_fixed_design_region(
        fixed_design_region,
        dimension,
        coordinate,
        polar_angles_in_degrees=True,
    )
    _validate_dimension_coordinate(dimension, coordinate)
    # organize argumant
    _design_region = deepcopy(design_region)
    if dimension == "2D":
        _design_region = [list(axis) for axis in _to_2d_region(_design_region)]  # type: ignore[arg-type]
    else:
        _design_region = [list(axis) for axis in _to_3d_region(_design_region)]  # type: ignore[arg-type]
    if coordinate == "Polar":
        _design_region[1][0] = np.deg2rad(_design_region[1][0])
        _design_region[1][1] = np.deg2rad(_design_region[1][1])
    # arrange NGnet
    mu = ngnet.arrange_gaussian_bases(
        sigma,
        _design_region,
        dimension,
        coordinate,
        distance_factor,
        eliminate_bases_on_edge=eliminate_bases_on_edge,
    )
    ngnet.set_parameters(mu, sigma)
    if dimension == "2D":
        ngnet.visualize_gaussian_bases_radius(
            _design_region,
            coordinate,
            "gaussian.png",
            check_basis_GUI=check_basis_GUI,
        )
    else:
        ngnet.visualize_gaussian_bases_radius_3d(
            _design_region,
            coordinate,
            "gaussian_3d.png",
            check_basis_GUI=check_basis_GUI,
        )
    return ngnet
