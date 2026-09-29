"""
simple_level_set_radius.py
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

import numpy as np

from emsopt_engine.interface.level_set_function_interface import LevelSetFunctionInterface
from emsopt_engine.registry import level_set_function
from utils.visualization import draw_3d_surface


class SimpleLevelSetRadius(LevelSetFunctionInterface):
    def __init__(self, num_level: int, scale: float = 1.0) -> None:
        self.num_level = num_level
        self.levels = np.linspace(1, -1, num_level)
        self.scale = scale

    def get_variable_dimension(self) -> int:
        """Get variable dimension

        Returns:
            int: dimension
        """
        return self.num_level - 1

    def calculate_output(self, parameters: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Batch compute the output of the level set function defined by parameters at points

        Args:
            parameters (np.ndarray): K-dimensional vector (K: number of parameters of the level set function)
            points (np.ndarray): 2D array (N points * D dimensions), each row corresponds to a calculation point

        Returns:
            Output calculation result (N-dimensional vector)
        """
        solution_cumul = np.cumsum(parameters) * self.scale
        points_radius = np.linalg.norm(points, axis=1)
        # level set
        points_level = np.zeros_like(points_radius)
        points_level[points_radius < solution_cumul[0]] = self.levels[0]
        for level_idx in range(self.num_level - 1):
            points_level[points_radius > solution_cumul[level_idx]] = self.levels[level_idx + 1]
        return points_level

    def visualize_profile_2d(
        self,
        solution: np.ndarray,
        num_division: int = 100,
    ) -> None:
        """
        Visualize the level set result as a 3D surface plot.
        Args:
            solution (array-like): Solution vector.
            num_division (int): Number of grid divisions per axis.
        """
        x_max = y_max = np.max(solution) * 2.0
        # Generate grid data
        x = np.linspace(0, x_max, num_division)
        y = np.linspace(0, y_max, num_division)
        X, Y = np.meshgrid(x, y)
        # Create list of (x, y) tuples
        xy_tuples = np.column_stack((X.ravel(), Y.ravel()))
        # Define function and calculate Z (z = f(x, y))
        z = self.calculate_output(solution, xy_tuples)
        Z = z.reshape((num_division, num_division))
        draw_3d_surface(X, Y, Z)


@level_set_function("ls_r")
def build_ls_r(num_level: int, scale: float = 1.0) -> LevelSetFunctionInterface:
    """Simple level set function whose bound is defined by radius in coordinate system.

    Example:
        If num_level = 3, variables are [r1, r2]
        The level is set like below (where r is radius of a coordinate):
            if r < r1, level 0 is set
            if r > r1 and r < r1 + r2, level 1 is set
            if r > r1 + r2, level 2 is set
        The definition of variables are differential (NOT definite radius from origin).

    Args:
        num_level (int): number of levels.
        scale (float): scale for solution.
    """
    return SimpleLevelSetRadius(num_level, scale)
