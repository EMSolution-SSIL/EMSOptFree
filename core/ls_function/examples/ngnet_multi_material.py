"""
ngnet_multi_material.py
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

from copy import deepcopy
from typing import Literal

import matplotlib as mpl

# for stable image output
mpl.use("Agg")

import numpy as np

from core.ls_function.examples.ngnet import NGnet
from emsopt_engine.interface.level_set_function_interface import LevelSetFunctionInterface
from emsopt_engine.registry import level_set_function


class NGnetMultiMaterial(NGnet):
    def __init__(self, normalize_output: bool = True, angle_1: float = 120.0, angle_2: float = 120.0) -> None:
        super().__init__(normalize_output)
        self.angle_1 = angle_1
        self.angle_2 = angle_2

    def get_variable_dimension(self) -> int:
        return self.K * 2

    def calculate_output(self, parameters: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Batch compute the output of NGnet
        Args:
            parameters (np.ndarray): Weight coefficients
            points (np.ndarray): Matrix of calculation points
        Returns:
            np.ndarray: Output calculation result
        """
        normalized_gaussian = self.calculate_ngnet(points).T
        ngnet_1 = np.dot(parameters[: self.K], normalized_gaussian)
        ngnet_2 = np.dot(parameters[self.K : 2 * self.K], normalized_gaussian)
        angles = np.degrees(np.arctan2(ngnet_2, ngnet_1))
        levels = np.zeros_like(angles)
        # starting from -180.0 deg, assign three levels
        levels[angles < -180.0 + self.angle_1] = 1
        levels[angles >= -180.0 + self.angle_1] = 0
        levels[angles >= -180.0 + self.angle_1 + self.angle_2] = -1
        return levels


@level_set_function("ngnet_multi_material")
def build_ngnet(
    coordinate: Literal["Cartesian", "Polar"],
    sigma: float,
    design_region: list[list[float, float], list[float, float]],
    distance_factor: float = 0.8,
    eliminate_bases_on_edge: bool = False,
    normalize_output: bool = True,
    angle_1: float = 120.0,
    angle_2: float = 120.0,
) -> LevelSetFunctionInterface:
    """
    NGnet class. Assumes the covariance matrix has only diagonal components.

    Args:
        coordinate (str): Coordinate system. "Cartesian" or "Polar".
        sigma (float): Standard deviation of the Gaussian basis functions to be arranged
        design_region (list[list[float, float], list[float, float]]): design region definition.
            [[x1, x2], [y1, y2]] for Cartesian, or [[r1, r2], [theta1, theta2]] for Polar.
        distance_factor: Value indicating how far apart the centers of the Gaussian basis functions.
            Please refer to NGnet.arrange_gaussian_bases method. Defaults to 0.8.
        eliminate_bases_on_edge (bool, optional): Whether to remove basis functions located on the region boundary.
            Defaults to False.
        normalize_output (bool): If False, normalization of function output will be disabled.
            (so it will be no longer "NGnet", but just weighted sum of Gaussian bases). Defaults to True.
        angle_1 (float): angle for first material on multimaterial map.
        angle_2 (float): angle for second material on multimaterial map.
    """
    ngnet = NGnetMultiMaterial(normalize_output, angle_1, angle_2)
    # organize argumant
    _design_region = deepcopy(design_region)
    if _design_region[0][0] > _design_region[0][1]:
        _design_region[0][0], _design_region[0][1] = _design_region[0][1], _design_region[0][0]
    if _design_region[1][0] > _design_region[1][1]:
        _design_region[1][0], _design_region[1][1] = _design_region[1][1], _design_region[1][0]
    if coordinate == "Polar":
        _design_region[1][0] = np.deg2rad(_design_region[1][0])
        _design_region[1][1] = np.deg2rad(_design_region[1][1])
    # arrange NGnet
    mu = ngnet.arrange_gaussian_bases(sigma, _design_region, coordinate, distance_factor, eliminate_bases_on_edge)
    ngnet.set_parameters(mu, sigma)
    ngnet.visualize_gaussian_bases_radius(_design_region, coordinate, "gaussian.png")
    return ngnet
