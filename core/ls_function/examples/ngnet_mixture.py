"""
ngnet_mixture.py
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

from typing import Literal

import numpy as np

from core.ls_function.examples.ngnet import NGnet
from core.ls_function.examples.simple_level_set_radius import SimpleLevelSetRadius
from emsopt_engine.interface.level_set_function_interface import LevelSetFunctionInterface
from emsopt_engine.registry import level_set_function


class NGnetMixture(LevelSetFunctionInterface):
    def __init__(self, ngnet: NGnet, bound_r: float, inversed: bool = False) -> None:
        self.ngnet = ngnet
        self.ls_r = SimpleLevelSetRadius(2)
        self.bound_r = bound_r
        self.inversed = inversed

    def get_variable_dimension(self) -> int:
        return self.ngnet.get_variable_dimension()

    def calculate_output(self, parameters: np.ndarray, points: np.ndarray) -> np.ndarray:
        # first, level set by radius
        level_radius = self.ls_r.calculate_output(self.bound_r, points)
        level = np.zeros_like(level_radius)
        if self.inversed:
            # level set points with r > bound_r, i.e. level_radius < 0
            level[level_radius < 0] = self.ngnet.calculate_output(parameters, points[level_radius < 0])
            # set on value (1) for points with r <= bound_r, i.e. level_radius >= 0
            level[level_radius >= 0] = 1
        else:
            # level set points with r < bound_r, i.e. level_radius > 0
            level[level_radius > 0] = self.ngnet.calculate_output(parameters, points[level_radius > 0])
            # set on value (1) for points with r >= bound_r, i.e. level_radius <= 0
            level[level_radius <= 0] = 1
        return level


@level_set_function("ngnet_mixture")
def build_ngnet_mixture(
    sigma: float,
    design_region: tuple[tuple[float, float], tuple[float, float]],
    coordinate: Literal["Cartesian", "Polar"],
    boundary_r: float,
    inversed: bool = False,
    distance_factor: float = 0.8,
    eliminate_bases_on_edge: bool = False,
    normalize_output: bool = True,
) -> LevelSetFunctionInterface:
    """
    NGnet_mixture class. Assumes the covariance matrix has only diagonal components.
    Level is set by the following procedure:
        1: entire region is splitted by boundary_r
        2: for r < bound_r (or r > bound_r when inversed), NGnet is applied

    Args:
        sigma (float): Standard deviation of the Gaussian basis functions to be arranged
        design_region (PhysicalRegion): design region definition.
            ((x1, x2), (y1, y2)) for Cartesian, or ((r1, r2), (theta1, theta2)) for Polar.
        coordinate (str): Coordinate system. "Cartesian" or "Polar".
        boundary_r (float): boundary radius to split level set region
        inversed (bool): inverse region where NGnet is applied
        distance_factor: Value indicating how far apart the centers of the Gaussian basis functions.
            Please refer to NGnet.arrange_gaussian_bases method. Defaults to 0.8.
        eliminate_bases_on_edge (bool, optional): Whether to remove basis functions located on the region boundary.
            Defaults to False.
        normalize_output (bool): If False, normalization of function output will be disabled.
        (so it will be no longer "NGnet", but just weighted sum of Gaussian bases). Defaults to True.
    """
    ngnet = NGnet(normalize_output)
    # organize argumant
    if design_region[0][0] > design_region[0][1]:
        design_region[0][0], design_region[0][1] = design_region[0][1], design_region[0][0]
    if design_region[1][0] > design_region[1][1]:
        design_region[1][0], design_region[1][1] = design_region[1][1], design_region[1][0]
    if coordinate == "Polar":
        design_region[1][0] = np.deg2rad(design_region[1][0])
        design_region[1][1] = np.deg2rad(design_region[1][1])
    # arrange NGnet
    mu = ngnet.arrange_gaussian_bases(sigma, design_region, coordinate, distance_factor, eliminate_bases_on_edge)
    ngnet.set_parameters(mu, sigma)
    ngnet.visualize_gaussian_bases_radius(design_region, coordinate, "gaussian.png")
    return NGnetMixture(ngnet, boundary_r, inversed)
