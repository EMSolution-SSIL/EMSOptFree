"""
ngnet.py
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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import lines, patches

from emsopt_engine.interface.level_set_function_interface import LevelSetFunctionInterface
from emsopt_engine.registry import level_set_function
from utils.visualization import draw_3d_surface


class NGnet(LevelSetFunctionInterface):
    def __init__(self, normalize_output: bool = True) -> None:
        """
        Initialize parameters
        """
        self.normalize_output = normalize_output
        self.K = 0  # Number of Gaussian basis functions
        self.D = 0  # Dimension of coordinate space
        self.mu = None  # (K, D)
        self.sigma = None  # (K,) - scalar values

    def get_variable_dimension(self) -> int:
        return self.K

    def set_parameters(self, mu: np.ndarray, sigma: np.ndarray) -> None:
        """Set parameters (mean, standard deviation)
        Args:
            mu: List of mean vectors (2D array of K*D)
            sigma: List of standard deviations (K-dimensional vector)
        """
        self.mu = np.array(mu)
        self.sigma = np.array(sigma)
        self.K, self.D = self.mu.shape
        self._reshape_mu()
        self._reshape_sigma()
        self._validate_dimensions()

    def load_parameters_from_csv(self, mu_filepath: str, sigma_filepath: str) -> None:
        """Load parameters from csv files and set them
        Args:
            mu_file (str): File path for mean vector data
            sigma_file (str): File path for standard deviation data
        """
        mu = np.loadtxt(mu_filepath, delimiter=",")
        sigma = np.loadtxt(sigma_filepath, delimiter=",")
        self.set_parameters(mu, sigma)

    def arrange_gaussian_bases(
        self,
        sigma: float,
        design_region: list[list[float, float], list[float, float]],
        coordinate: Literal["Cartesian", "Polar"],
        distance_factor: float = 0.8,
        eliminate_bases_on_edge: bool = False,
    ) -> np.ndarray:
        """Arrange Gaussian basis functions within the region
        - Standard deviation is set to the given sigma
        - The centers are determined to be densely and uniformly distributed within the region
        - Only supports 2D rectangular or sector regions

        Args:
            sigma (float): Standard deviation of the Gaussian basis functions to be arranged
            design_region (PhysicalRegion): design region definition ((x1, x2), (y1, y2)) or ((r1, r2), (t1, t2))
            coordinate (str, optional): Coordinate system. "Cartesian" or "Polar".
            distance_factor (float, optional): Recommended around 0.8 (0.7~0.9).
                Value indicating how far apart the centers of the Gaussian basis functions.
                If 1.0, the centers are separated by 2*sigma.
                This is the distance at which the Gaussian basis functions barely do not overlap (with some deviation).
                The smaller this value, the closer the centers, and thus the more overlap.
            eliminate_bases_on_edge (bool, optional): Whether to remove basis functions located on the region boundary.
                Defaults to False.
        """
        if distance_factor <= 0:
            msg = f"distance_factor {distance_factor}, should be positive"
            raise ValueError(msg)

        mu = []
        center_point = [0, 0]
        distance_between_nearest = 2 * sigma * distance_factor
        # assign a-axis values
        num_step_a = (design_region[0][1] - design_region[0][0]) / distance_between_nearest
        a_values = np.linspace(design_region[0][0], design_region[0][1], int(np.round(num_step_a)) + 1)
        for i, a in enumerate(a_values):
            if eliminate_bases_on_edge and ((i == 0) or (i == len(a_values) - 1)):
                continue
            center_point[0] = a
            # assign b-axis values
            if coordinate == "Cartesian":
                num_step_b = (design_region[1][1] - design_region[1][0]) / distance_between_nearest
            elif coordinate == "Polar":
                # calculate number of steps which equally divide arc with radius a
                num_step_b = a * (design_region[1][1] - design_region[1][0]) / distance_between_nearest
            b_values = np.linspace(design_region[1][0], design_region[1][1], int(np.round(num_step_b)) + 1)
            for j, b in enumerate(b_values):
                if eliminate_bases_on_edge and ((j == 0) or (j == len(b_values) - 1)):
                    continue
                center_point[1] = b
                mu.append(deepcopy(center_point))
        mu = np.array(mu)
        if coordinate == "Polar":
            # Convert mu into Cartesian
            for i in range(len(mu)):
                mu[i] = np.array([mu[i, 0] * np.cos(mu[i, 1]), mu[i, 0] * np.sin(mu[i, 1])])
        return mu

    def _reshape_mu(self) -> None:
        if self.mu.ndim == 1:
            self.mu = self.mu.reshape(1, -1)

    def _reshape_sigma(self) -> None:
        if self.sigma.ndim == 0:
            self.sigma = np.array([self.sigma] * self.K)
        elif self.sigma.ndim > 1:
            self.sigma = self.sigma.flatten()

    def _validate_dimensions(self) -> None:
        if self.sigma.shape[0] != self.K:
            msg = f"Inconsistent dimensions: mu has {self.K} components, but sigma has {self.sigma.shape[0]}"
            raise ValueError(msg)

    def gaussian(self, x: np.ndarray, k: int) -> float:
        """
        Compute the k-th Gaussian basis function.

        Args:
            x (np.ndarray): Input vector (D-dimensional)
            k (int): Index of the Gaussian basis function

        Returns:
            float: Value of the k-th Gaussian basis function at x

        Raises:
            RuntimeError: If parameters (mu and sigma) are not set.
            ValueError: If the dimension of x does not match D.
        """
        if self.mu is None or self.sigma is None:
            msg = "Parameters (mu and sigma) not loaded."
            raise RuntimeError(msg)
        if x.shape[0] != self.D:
            msg = f"Input dimension {x.shape[0]} does not match expected {self.D}."
            raise ValueError(msg)

        diff = x - self.mu[k]
        exponent = -0.5 * np.dot(diff, diff) / np.power(self.sigma[k], 2)
        denom = np.power(2 * np.pi * np.power(self.sigma[k], 2), self.D / 2)
        return float(np.exp(exponent) / denom)

    def calculate_ngnet(self, X: np.ndarray) -> np.ndarray:
        """
        Compute normalized gating values φ_k(x_n) for multiple input points.

        Args:
            X (np.ndarray): Shape (N, D), where N is the number of input vectors.

        Returns:
            np.ndarray:
                Shape (N, K), where each row contains normalized Gaussian weights for one input.

        Raises:
            RuntimeError: If parameters (mu and sigma) are not set.
            ValueError: If the dimension of input vectors does not match D.
        """
        if self.mu is None or self.sigma is None:
            msg = "Parameters (mu and sigma) not loaded."
            raise RuntimeError(msg)

        _X = np.atleast_2d(X)
        N, D_in = _X.shape
        if D_in != self.D:
            msg = f"Input dimension {D_in} does not match expected {self.D}."
            raise ValueError(msg)

        # (N, 1, D) - input broadcasted
        # (1, K, D) - mu broadcasted
        diff = _X[:, np.newaxis, :] - self.mu[np.newaxis, :, :]  # Shape: (N, K, D)

        # squared norm ||x - mu||^2 for each (n,k)
        squared_dist = np.sum(diff**2, axis=2)  # Shape: (N, K)

        # denominator part from scalar σ_k (1 × K)
        sigma = self.sigma[np.newaxis, :]  # Shape: (1, K)

        # log φ_k(x) = -0.5 * ||x-mu||^2 / σ^2 - (D/2)*log(2πσ^2)
        log_phi = -0.5 * squared_dist / (sigma**2)
        log_phi -= (self.D / 2) * np.log(2 * np.pi * sigma**2)

        # === log-sum-exp trick for safe exp ===
        max_log_phi = np.max(log_phi, axis=1, keepdims=True)  # Shape: (N, 1)
        log_phi_shifted = log_phi - max_log_phi
        phi_unnorm = np.exp(log_phi_shifted)

        # normalization if enabled
        phi = phi_unnorm / np.sum(phi_unnorm, axis=1, keepdims=True) if self.normalize_output else phi_unnorm

        return phi

    def calculate_output(self, parameters: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Batch compute the output of NGnet
        Args:
            parameters (np.ndarray): Weight coefficients
            points (np.ndarray): Matrix of calculation points
        Returns:
            np.ndarray: Output calculation result
        """
        return np.dot(parameters, self.calculate_ngnet(points).T)

    def visualize_gaussian_bases_radius(
        self,
        design_region: list[list[float, float], list[float, float]],
        coordinate: Literal["Cartesian", "Polar"],
        filepath: str,
        no_axis: bool = False,
    ) -> None:
        """
        Visualize the radii of Gaussian bases as circles on a 2D plot.
        Args:
            design_region (PhysicalRegion): design region definition ((x1, x2), (y1, y2)) or ((r1, r2), (t1, t2))
            coordinate (str, optional): Coordinate system. "Cartesian" or "Polar".
            filepath (str): filepath to save plot
            no_axis (bool): If True, hides axis and title.
        """
        # Prepare figure and axis
        fig, ax = plt.subplots()
        # Draw circles
        for k in range(self.K):
            circle = patches.Circle(self.mu[k], self.sigma[k], edgecolor="black", facecolor="none", linewidth=1)
            ax.add_patch(circle)
        # Draw design region
        if coordinate == "Cartesian":
            idx = [(0, 0), (0, 1), (1, 1), (1, 0)]
            for i in range(4):
                line = lines.Line2D(
                    [design_region[0][idx[i % 4][0]], design_region[0][idx[(i + 1) % 4][0]]],
                    [design_region[1][idx[i % 4][1]], design_region[1][idx[(i + 1) % 4][1]]],
                    linewidth=1,
                    color="black",
                )
                ax.add_line(line)
        elif coordinate == "Polar":
            inner_arc = patches.Arc(
                (0, 0),
                2 * design_region[0][0],
                2 * design_region[0][0],
                theta1=np.rad2deg(design_region[1][0]),
                theta2=np.rad2deg(design_region[1][1]),
                linewidth=1,
                color="black",
            )
            ax.add_patch(inner_arc)
            outer_arc = patches.Arc(
                (0, 0),
                2 * design_region[0][1],
                2 * design_region[0][1],
                theta1=np.rad2deg(design_region[1][0]),
                theta2=np.rad2deg(design_region[1][1]),
                linewidth=1,
                color="black",
            )
            ax.add_patch(outer_arc)
            line_1 = lines.Line2D(
                [design_region[0][0] * np.cos(design_region[1][0]), design_region[0][1] * np.cos(design_region[1][0])],
                [design_region[0][0] * np.sin(design_region[1][0]), design_region[0][1] * np.sin(design_region[1][0])],
                linewidth=1,
                color="black",
            )
            ax.add_line(line_1)
            line_2 = lines.Line2D(
                [design_region[0][0] * np.cos(design_region[1][1]), design_region[0][1] * np.cos(design_region[1][1])],
                [design_region[0][0] * np.sin(design_region[1][1]), design_region[0][1] * np.sin(design_region[1][1])],
                linewidth=1,
                color="black",
            )
            ax.add_line(line_2)
        fig.patch.set_visible(False)

        if no_axis:
            ax.set_title("")
            ax.axis("off")
        else:
            plt.title("Gaussian bases")

        ax.set_aspect("equal")
        plt.autoscale()
        plt.savefig(filepath)
        plt.close()

    def visualize_gaussian_bases(
        self,
        filepath: str,
        num_division: int = 100,
        x1: float = -1,
        x2: float = 1,
        y1: float = -1,
        y2: float = 1,
    ) -> None:
        """
        Visualize the sum of Gaussian bases as a 3D surface plot.

        Args:
            filepath (str): filepath to save plot
            num_division (int): Number of grid divisions per axis.
            x1, x2, y1, y2 (float): Axis limits.
        """
        # Generate grid data
        x = np.linspace(x1, x2, num_division)
        y = np.linspace(y1, y2, num_division)
        X, Y = np.meshgrid(x, y)
        # Create list of (x, y) tuples
        xy_tuples = np.column_stack((X.ravel(), Y.ravel()))
        # Define function and calculate Z (z = f(x, y))
        z = np.zeros(xy_tuples.shape[0])
        for k in range(self.K):
            for i in range(len(xy_tuples)):
                z[i] += self.gaussian(xy_tuples[i], k)
        Z = z.reshape((num_division, num_division))
        draw_3d_surface(X, Y, Z, filepath)

    def visualize_profile_2d(
        self,
        filepath: str,
        weight: np.ndarray,
        num_division: int = 100,
        x1: float = -1,
        x2: float = 1,
        y1: float = -1,
        y2: float = 1,
    ) -> None:
        """
        Visualize the weighted sum of NGnet bases as a 3D surface plot.

        Args:
            filepath (str): filepath to save plot
            weight (array-like): Weights for each base.
            num_division (int): Number of grid divisions per axis.
            x1, x2, y1, y2 (float): Axis limits.
        """
        # Generate grid data
        x = np.linspace(x1, x2, num_division)
        y = np.linspace(y1, y2, num_division)
        X, Y = np.meshgrid(x, y)
        # Create list of (x, y) tuples
        xy_tuples = np.column_stack((X.ravel(), Y.ravel()))
        # Define function and calculate Z (z = f(x, y))
        bases = self.calculate_ngnet(xy_tuples)
        z = np.dot(weight, bases.T)
        Z = z.reshape((num_division, num_division))
        draw_3d_surface(X, Y, Z, filepath)


@level_set_function("ngnet")
def build_ngnet(
    coordinate: Literal["Cartesian", "Polar"],
    sigma: float,
    design_region: list[list[float, float], list[float, float]],
    distance_factor: float = 0.8,
    eliminate_bases_on_edge: bool = False,
    normalize_output: bool = True,
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
    """
    ngnet = NGnet(normalize_output)
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
