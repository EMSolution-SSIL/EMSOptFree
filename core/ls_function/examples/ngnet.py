"""
ngnet.py
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

# ruff : noqa: C901, PLR0912, PLR0915, PLR2004
# Dimension and plotting constants are intentionally literal in this module.
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import lines, patches

from emsopt_engine.interface.level_set_function_interface import LevelSetFunctionInterface
from emsopt_engine.registry import level_set_function
from utils.visualization import draw_3d_surface

Dimension = Literal["2D", "3D"]
CoordinateSystem = Literal["Cartesian", "Polar"]
FULL_ANGLE_RADIANS = 2.0 * np.pi


@dataclass(frozen=True)
class NGnetParameters:
    mu: np.ndarray
    sigma: np.ndarray
    normalize_output: bool = True


@dataclass(frozen=True)
class GridArrangeSpec2D:
    sigma: float
    design_region: tuple[tuple[float, float], tuple[float, float]]
    coordinate: CoordinateSystem
    distance_factor: float = 0.8
    eliminate_bases_on_edge: bool = False


@dataclass(frozen=True)
class GridArrangeSpec3D:
    sigma: float
    design_region: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    coordinate: CoordinateSystem
    distance_factor: float = 0.8
    eliminate_bases_on_edge: bool = False


class NGnetError(Exception):
    """Base exception for NGnet operations."""


class NGnetValidationError(NGnetError, ValueError):
    """Input data is invalid and must be corrected."""


class NGnetStateError(NGnetError, RuntimeError):
    """Required model state is missing."""


class NGnetNumericalError(NGnetError, ArithmeticError):
    """Numerical instability occurred during computation."""


class NGnetIOError(NGnetError, OSError):
    """I/O operation for NGnet parameters failed."""


def _validate_dimension_coordinate(dimension: str, coordinate: str) -> None:
    allowed = {("2D", "Cartesian"), ("2D", "Polar"), ("3D", "Cartesian"), ("3D", "Polar")}
    if (dimension, coordinate) not in allowed:
        msg = f"Unsupported combination: dimension={dimension}, coordinate='{coordinate}'."
        raise NGnetValidationError(msg)


def _dimension_size(dimension: Dimension) -> int:
    if dimension == "2D":
        return 2
    if dimension == "3D":
        return 3
    msg = f"Unsupported dimension '{dimension}'."
    raise NGnetValidationError(msg)


def _normalize_fixed_design_region(
    fixed_design_region: tuple | list | None,
    dimension: Dimension,
    coordinate: CoordinateSystem,
    *,
    polar_angles_in_degrees: bool = False,
) -> tuple[tuple[float, float], ...] | None:
    if fixed_design_region is None:
        return None
    _validate_dimension_coordinate(dimension, coordinate)
    dimension_size = _dimension_size(dimension)
    try:
        region = np.asarray(fixed_design_region, dtype=float)
    except (TypeError, ValueError) as exc:
        msg = f"fixed_design_region must be a valid {dimension} region."
        raise NGnetValidationError(msg) from exc
    if region.shape != (dimension_size, 2) or not np.all(np.isfinite(region)):
        msg = f"fixed_design_region must be a finite {dimension} region."
        raise NGnetValidationError(msg)
    region = region.copy()
    if coordinate == "Polar":
        region[0].sort()
        if polar_angles_in_degrees:
            region[1] = np.deg2rad(region[1])
        if dimension == "3D":
            region[2].sort()
    else:
        region.sort(axis=1)
    return tuple((float(axis[0]), float(axis[1])) for axis in region)


def _to_2d_region(
    design_region: tuple[tuple[float, float], tuple[float, float]] | list[list[float]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    if len(design_region) != 2:
        msg = f"2D design_region must have 2 axes, but got {len(design_region)}."
        raise NGnetValidationError(msg)
    a1, a2 = design_region[0]
    b1, b2 = design_region[1]
    return ((min(a1, a2), max(a1, a2)), (min(b1, b2), max(b1, b2)))


def _to_3d_region(
    design_region: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | list[list[float]],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    if len(design_region) != 3:
        msg = f"3D design_region must have 3 axes, but got {len(design_region)}."
        raise NGnetValidationError(msg)
    x1, x2 = design_region[0]
    y1, y2 = design_region[1]
    z1, z2 = design_region[2]
    return ((min(x1, x2), max(x1, x2)), (min(y1, y2), max(y1, y2)), (min(z1, z2), max(z1, z2)))


def _axis_values(start: float, end: float, distance: float) -> np.ndarray:
    num_step = (end - start) / distance
    return np.linspace(start, end, int(np.round(num_step)) + 1)


def _validate_arrange_inputs(sigma: float, distance_factor: float) -> None:
    if sigma <= 0:
        msg = f"sigma must be positive, but got {sigma}."
        raise NGnetValidationError(msg)
    if distance_factor <= 0:
        msg = f"distance_factor {distance_factor}, should be positive."
        raise NGnetValidationError(msg)


def _is_edge_index(index: int, length: int) -> bool:
    return index == 0 or index == length - 1


def _num_step_b_for_2d(
    coordinate: CoordinateSystem,
    a_value: float,
    b_start: float,
    b_end: float,
    distance_between_nearest: float,
) -> float:
    if coordinate == "Cartesian":
        return (b_end - b_start) / distance_between_nearest
    if coordinate == "Polar":
        return a_value * (b_end - b_start) / distance_between_nearest
    msg = f"Unsupported coordinate '{coordinate}' for 2D arrangement."
    raise NGnetValidationError(msg)


def _num_step_theta_for_3d_polar(
    radius_value: float,
    theta_start: float,
    theta_end: float,
    distance_between_nearest: float,
) -> float:
    return radius_value * (theta_end - theta_start) / distance_between_nearest


def _convert_polar_3d_to_cartesian(mu: np.ndarray) -> np.ndarray:
    radius = mu[:, 0]
    theta = mu[:, 1]
    z = mu[:, 2]
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta), z))


def _finalize_basis_plot(
    fig: mpl.figure.Figure,
    filepath: str,
    *,
    check_basis_GUI: bool,
) -> None:
    if check_basis_GUI:
        plt.show(block=True)
    else:
        plt.savefig(filepath)
    plt.close(fig)


def _build_sphere_surface(
    center: np.ndarray,
    radius: float,
    *,
    sphere_resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0.0, np.pi, sphere_resolution)
    v = np.linspace(0.0, 2.0 * np.pi, sphere_resolution)
    uu, vv = np.meshgrid(u, v)
    x = center[0] + radius * np.sin(uu) * np.cos(vv)
    y = center[1] + radius * np.sin(uu) * np.sin(vv)
    z = center[2] + radius * np.cos(uu)
    return x, y, z


def arrange_gaussian_bases_2d(spec: GridArrangeSpec2D) -> np.ndarray:
    _validate_arrange_inputs(spec.sigma, spec.distance_factor)

    region = _to_2d_region(spec.design_region)
    distance_between_nearest = 2 * spec.sigma * spec.distance_factor
    a_values = _axis_values(region[0][0], region[0][1], distance_between_nearest)
    mu: list[list[float]] = []

    for i, a in enumerate(a_values):
        if spec.eliminate_bases_on_edge and _is_edge_index(i, len(a_values)):
            continue
        num_step_b = _num_step_b_for_2d(
            spec.coordinate,
            a,
            region[1][0],
            region[1][1],
            distance_between_nearest,
        )
        b_values = np.linspace(region[1][0], region[1][1], int(np.round(num_step_b)) + 1)
        for j, b in enumerate(b_values):
            if spec.eliminate_bases_on_edge and _is_edge_index(j, len(b_values)):
                continue
            mu.append([a, b])

    mu_arr = np.array(mu, dtype=float)
    if mu_arr.size == 0:
        return mu_arr.reshape(0, 2)
    if spec.coordinate == "Polar":
        r = mu_arr[:, 0]
        theta = mu_arr[:, 1]
        mu_arr = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    return mu_arr


def arrange_gaussian_bases_3d(spec: GridArrangeSpec3D) -> np.ndarray:
    _validate_arrange_inputs(spec.sigma, spec.distance_factor)

    region = _to_3d_region(spec.design_region)
    distance_between_nearest = 2 * spec.sigma * spec.distance_factor
    z_values = _axis_values(region[2][0], region[2][1], distance_between_nearest)

    mu: list[list[float]] = []
    if spec.coordinate == "Cartesian":
        x_values = _axis_values(region[0][0], region[0][1], distance_between_nearest)
        y_values = _axis_values(region[1][0], region[1][1], distance_between_nearest)
        for ix, x in enumerate(x_values):
            if spec.eliminate_bases_on_edge and _is_edge_index(ix, len(x_values)):
                continue
            for iy, y in enumerate(y_values):
                if spec.eliminate_bases_on_edge and _is_edge_index(iy, len(y_values)):
                    continue
                for iz, z in enumerate(z_values):
                    if spec.eliminate_bases_on_edge and _is_edge_index(iz, len(z_values)):
                        continue
                    mu.append([x, y, z])
    elif spec.coordinate == "Polar":
        radius_values = _axis_values(region[0][0], region[0][1], distance_between_nearest)
        for ir, radius in enumerate(radius_values):
            if spec.eliminate_bases_on_edge and _is_edge_index(ir, len(radius_values)):
                continue
            num_step_theta = _num_step_theta_for_3d_polar(
                radius,
                region[1][0],
                region[1][1],
                distance_between_nearest,
            )
            theta_values = np.linspace(region[1][0], region[1][1], int(np.round(num_step_theta)) + 1)
            for itheta, theta in enumerate(theta_values):
                if spec.eliminate_bases_on_edge and _is_edge_index(itheta, len(theta_values)):
                    continue
                for iz, z in enumerate(z_values):
                    if spec.eliminate_bases_on_edge and _is_edge_index(iz, len(z_values)):
                        continue
                    mu.append([radius, theta, z])
    else:
        msg = f"Unsupported coordinate '{spec.coordinate}' for 3D arrangement."
        raise NGnetValidationError(msg)

    mu_arr = np.array(mu, dtype=float)
    if mu_arr.size == 0:
        return mu_arr.reshape(0, 3)
    if spec.coordinate == "Polar":
        return _convert_polar_3d_to_cartesian(mu_arr)
    return mu_arr


def load_parameters_from_csv(mu_filepath: str, sigma_filepath: str) -> NGnetParameters:
    try:
        mu = np.loadtxt(mu_filepath, delimiter=",")
        sigma = np.loadtxt(sigma_filepath, delimiter=",")
    except OSError as exc:
        msg = f"Failed to load parameter CSV files: mu='{mu_filepath}', sigma='{sigma_filepath}'."
        raise NGnetIOError(msg) from exc
    except ValueError as exc:
        msg = f"Failed to parse parameter CSV files: mu='{mu_filepath}', sigma='{sigma_filepath}'."
        raise NGnetValidationError(msg) from exc
    return NGnetParameters(mu=np.asarray(mu), sigma=np.asarray(sigma))


def save_parameters_to_csv(params: NGnetParameters, mu_filepath: str, sigma_filepath: str) -> None:
    try:
        np.savetxt(mu_filepath, np.asarray(params.mu), delimiter=",")
        np.savetxt(sigma_filepath, np.asarray(params.sigma), delimiter=",")
    except OSError as exc:
        msg = f"Failed to save parameter CSV files: mu='{mu_filepath}', sigma='{sigma_filepath}'."
        raise NGnetIOError(msg) from exc


class NGnetModel:
    def __init__(self, params: NGnetParameters) -> None:
        self.params = self._validate_parameters(params)

    @property
    def K(self) -> int:
        return int(self.params.mu.shape[0])

    @property
    def D(self) -> int:
        return int(self.params.mu.shape[1])

    @staticmethod
    def _validate_parameters(params: NGnetParameters) -> NGnetParameters:
        mu = np.asarray(params.mu, dtype=float)
        sigma = np.asarray(params.sigma, dtype=float)

        if mu.ndim == 1:
            mu = mu.reshape(1, -1)
        if mu.ndim != 2:
            msg = f"mu must be a 2D array with shape (K, D), but got shape {mu.shape}."
            raise NGnetValidationError(msg)
        if mu.shape[0] == 0:
            msg = "mu must contain at least one basis."
            raise NGnetValidationError(msg)

        if sigma.ndim == 0:
            sigma = np.array([sigma] * mu.shape[0], dtype=float)
        elif sigma.ndim > 1:
            sigma = sigma.flatten()

        if sigma.shape[0] != mu.shape[0]:
            msg = f"Inconsistent dimensions: mu has {mu.shape[0]} components, but sigma has {sigma.shape[0]}."
            raise NGnetValidationError(msg)
        if np.any(sigma <= 0):
            msg = "sigma must be positive for all bases."
            raise NGnetValidationError(msg)
        if not np.isfinite(mu).all() or not np.isfinite(sigma).all():
            msg = "mu and sigma must be finite."
            raise NGnetValidationError(msg)

        return NGnetParameters(mu=mu, sigma=sigma, normalize_output=params.normalize_output)

    def gaussian(self, x: np.ndarray, k: int) -> float:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim != 1:
            msg = f"Input x must be a 1D array with shape ({self.D},), but got shape {x_arr.shape}."
            raise NGnetValidationError(msg)
        if x_arr.shape[0] != self.D:
            msg = f"Input dimension {x_arr.shape[0]} does not match expected {self.D}."
            raise NGnetValidationError(msg)
        if k < 0 or k >= self.K:
            msg = f"Basis index k={k} is out of range [0, {self.K - 1}]."
            raise NGnetValidationError(msg)

        diff = x_arr - self.params.mu[k]
        sigma_k = self.params.sigma[k]
        exponent = -0.5 * np.dot(diff, diff) / (sigma_k**2)
        denom = np.power(2 * np.pi * (sigma_k**2), self.D / 2)
        return float(np.exp(exponent) / denom)

    def calculate_ngnet(self, points: np.ndarray) -> np.ndarray:
        x = np.asarray(points, dtype=float)
        x = np.atleast_2d(x)
        if x.shape[1] != self.D:
            msg = f"Input dimension {x.shape[1]} does not match expected {self.D}."
            raise NGnetValidationError(msg)

        diff = x[:, np.newaxis, :] - self.params.mu[np.newaxis, :, :]
        squared_dist = np.sum(diff**2, axis=2)
        sigma = self.params.sigma[np.newaxis, :]

        log_phi = -0.5 * squared_dist / (sigma**2)
        log_phi -= (self.D / 2) * np.log(2 * np.pi * sigma**2)

        if not np.isfinite(log_phi).all():
            msg = "Numerical instability detected while evaluating Gaussian bases."
            raise NGnetNumericalError(msg)

        max_log_phi = np.max(log_phi, axis=1, keepdims=True)
        phi_unnorm = np.exp(log_phi - max_log_phi)

        if not self.params.normalize_output:
            return phi_unnorm

        denom = np.sum(phi_unnorm, axis=1, keepdims=True)
        if np.any(denom <= 0):
            msg = "Normalization failed due to zero denominator."
            raise NGnetNumericalError(msg)
        return phi_unnorm / denom

    def calculate_output(self, weights: np.ndarray, points: np.ndarray) -> np.ndarray:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.shape[0] != self.K:
            msg = f"weights length {w.shape[0]} does not match the number of bases K={self.K}."
            raise NGnetValidationError(msg)
        return self.calculate_ngnet(points) @ w


class NGnet(LevelSetFunctionInterface):
    def __init__(
        self,
        *,
        normalize_output: bool = True,
        dimension: Dimension = "2D",
        coordinate: CoordinateSystem = "Cartesian",
        fixed_design_region: tuple | list | None = None,
    ) -> None:
        """
        Initialize parameters
        """
        self.normalize_output = normalize_output
        self.K = 0  # Number of Gaussian basis functions
        self.D = 0  # Dimension of coordinate space
        self.mu = None  # (K, D)
        self.sigma = None  # (K,) - scalar values
        self._model: NGnetModel | None = None
        self.configure_fixed_design_region(fixed_design_region, dimension, coordinate)

    def configure_fixed_design_region(
        self,
        fixed_design_region: tuple | list | None,
        dimension: Dimension,
        coordinate: CoordinateSystem,
        *,
        polar_angles_in_degrees: bool = False,
    ) -> None:
        """Configure the region where NGnet output is variable."""
        _validate_dimension_coordinate(dimension, coordinate)
        self.dimension = dimension
        self.coordinate = coordinate
        self.fixed_design_region = _normalize_fixed_design_region(
            fixed_design_region,
            dimension,
            coordinate,
            polar_angles_in_degrees=polar_angles_in_degrees,
        )

    @staticmethod
    def _angles_inside_range(theta: np.ndarray, theta_min: float, theta_max: float) -> np.ndarray:
        if abs(theta_max - theta_min) >= FULL_ANGLE_RADIANS:
            return np.ones(theta.shape, dtype=bool)
        lower = theta_min % FULL_ANGLE_RADIANS
        upper = theta_max % FULL_ANGLE_RADIANS
        if lower <= upper:
            return (lower <= theta) & (theta <= upper)
        return (lower <= theta) | (theta <= upper)

    def _points_inside_fixed_design_region(self, points: np.ndarray) -> np.ndarray:
        points_array = np.asarray(points, dtype=float)
        if self.fixed_design_region is None:
            return np.ones(len(points_array), dtype=bool)
        expected_dimension = _dimension_size(self.dimension)
        if points_array.ndim != 2 or points_array.shape[1] != expected_dimension:
            actual_dimension = points_array.shape[1] if points_array.ndim == 2 else None
            msg = f"Input dimension {actual_dimension} does not match expected {expected_dimension}."
            raise NGnetValidationError(msg)

        if self.coordinate == "Cartesian":
            inside = np.ones(len(points_array), dtype=bool)
            for axis_idx, (axis_min, axis_max) in enumerate(self.fixed_design_region):
                inside &= (axis_min <= points_array[:, axis_idx]) & (points_array[:, axis_idx] <= axis_max)
            return inside

        radius = np.linalg.norm(points_array[:, :2], axis=1)
        theta = np.mod(np.arctan2(points_array[:, 1], points_array[:, 0]), FULL_ANGLE_RADIANS)
        radius_min, radius_max = self.fixed_design_region[0]
        theta_min, theta_max = self.fixed_design_region[1]
        inside = (radius_min <= radius) & (radius <= radius_max)
        inside &= self._angles_inside_range(theta, theta_min, theta_max)
        if self.dimension == "3D":
            z_min, z_max = self.fixed_design_region[2]
            inside &= (z_min <= points_array[:, 2]) & (points_array[:, 2] <= z_max)
        return inside

    def _calculate_output_in_fixed_design_region(
        self,
        points: np.ndarray,
        calculate: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        points_array = np.atleast_2d(np.asarray(points, dtype=float))
        variable_mask = self._points_inside_fixed_design_region(points_array)
        output = np.ones(len(points_array), dtype=float)
        if np.any(variable_mask):
            output[variable_mask] = calculate(points_array[variable_mask])
        return output

    def get_variable_dimension(self) -> int:
        return self.K

    def set_parameters(self, mu: np.ndarray, sigma: np.ndarray) -> None:
        """Set parameters (mean, standard deviation)
        Args:
            mu: List of mean vectors (2D array of K*D)
            sigma: List of standard deviations (K-dimensional vector)
        """
        params = NGnetParameters(mu=np.asarray(mu), sigma=np.asarray(sigma), normalize_output=self.normalize_output)
        self._model = NGnetModel(params)
        self.mu = self._model.params.mu
        self.sigma = self._model.params.sigma
        self.K = self._model.K
        self.D = self._model.D

    def load_parameters_from_csv(self, mu_filepath: str, sigma_filepath: str) -> None:
        """Load parameters from csv files and set them
        Args:
            mu_file (str): File path for mean vector data
            sigma_file (str): File path for standard deviation data
        """
        params = load_parameters_from_csv(mu_filepath, sigma_filepath)
        self.set_parameters(params.mu, params.sigma)

    def arrange_gaussian_bases(
        self,
        sigma: float,
        design_region: (
            tuple[tuple[float, float], tuple[float, float]]
            | tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
        ),
        dimension: Dimension,
        coordinate: CoordinateSystem,
        distance_factor: float = 0.8,
        *,
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
        _validate_dimension_coordinate(dimension, coordinate)
        if dimension == "2D":
            spec2d = GridArrangeSpec2D(
                sigma=sigma,
                design_region=_to_2d_region(design_region),  # type: ignore[arg-type]
                coordinate=coordinate,
                distance_factor=distance_factor,
                eliminate_bases_on_edge=eliminate_bases_on_edge,
            )
            return arrange_gaussian_bases_2d(spec2d)
        if dimension == "3D":
            spec3d = GridArrangeSpec3D(
                sigma=sigma,
                design_region=_to_3d_region(design_region),  # type: ignore[arg-type]
                coordinate=coordinate,
                distance_factor=distance_factor,
                eliminate_bases_on_edge=eliminate_bases_on_edge,
            )
            return arrange_gaussian_bases_3d(spec3d)
        msg = f"Unsupported dimension '{dimension}'."
        raise NGnetValidationError(msg)

    def _reshape_mu(self) -> None:
        if self.mu is None:
            return
        if self.mu.ndim == 1:
            self.mu = self.mu.reshape(1, -1)

    def _reshape_sigma(self) -> None:
        if self.sigma is None:
            return
        if self.sigma.ndim == 0:
            self.sigma = np.array([self.sigma] * self.K)
        elif self.sigma.ndim > 1:
            self.sigma = self.sigma.flatten()

    def _validate_dimensions(self) -> None:
        if self.mu is None or self.sigma is None:
            msg = "Parameters (mu and sigma) not loaded."
            raise NGnetStateError(msg)
        if self.sigma.shape[0] != self.K:
            msg = f"Inconsistent dimensions: mu has {self.K} components, but sigma has {self.sigma.shape[0]}"
            raise NGnetValidationError(msg)

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
        if self._model is None:
            msg = "Parameters (mu and sigma) not loaded."
            raise NGnetStateError(msg)
        return self._model.gaussian(x, k)

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
        if self._model is None:
            msg = "Parameters (mu and sigma) not loaded."
            raise NGnetStateError(msg)
        return self._model.calculate_ngnet(X)

    def calculate_output(self, weights: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Batch compute the output of NGnet
        Args:
            weights (np.ndarray): Weight coefficients
            points (np.ndarray): Matrix of calculation points
        Returns:
            np.ndarray: Output calculation result
        """
        if self._model is None:
            msg = "Parameters (mu and sigma) not loaded."
            raise NGnetStateError(msg)
        return self._calculate_output_in_fixed_design_region(
            points,
            lambda variable_points: self._model.calculate_output(weights, variable_points),
        )

    def visualize_gaussian_bases_radius(
        self,
        design_region: list[list[float, float], list[float, float]],
        coordinate: Literal["Cartesian", "Polar"],
        filepath: str,
        *,
        check_basis_GUI: bool = False,
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
        _finalize_basis_plot(fig, filepath, check_basis_GUI=check_basis_GUI)

    def visualize_gaussian_bases_radius_3d(
        self,
        design_region: list[list[float, float], list[float, float], list[float, float]],
        coordinate: CoordinateSystem,
        filepath: str,
        *,
        check_basis_GUI: bool = False,
        sphere_resolution: int = 20,
        render_style: Literal["wireframe", "surface"] = "wireframe",
        no_axis: bool = False,
    ) -> None:
        """
        Visualize 3D Gaussian basis radii as spheres with radius=sigma.
        Args:
            design_region: [[x1, x2], [y1, y2], [z1, z2]]
            filepath: filepath to save plot
            check_basis_GUI: If True, show GUI instead of saving image.
            sphere_resolution: Number of mesh divisions for each sphere.
            render_style: Sphere drawing style, "wireframe" or "surface".
            no_axis: If True, hides axis and title.
        """
        if self.mu is None or self.sigma is None:
            msg = "Parameters (mu and sigma) not loaded."
            raise NGnetStateError(msg)
        if self.D != 3:
            msg = f"visualize_gaussian_bases_radius_3d requires D=3, but got D={self.D}."
            raise NGnetValidationError(msg)
        if sphere_resolution < 4:
            msg = f"sphere_resolution must be >= 4, but got {sphere_resolution}."
            raise NGnetValidationError(msg)
        if render_style not in ("wireframe", "surface"):
            msg = f"Unsupported render_style '{render_style}'."
            raise NGnetValidationError(msg)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        for k in range(self.K):
            x, y, z = _build_sphere_surface(
                self.mu[k],
                float(self.sigma[k]),
                sphere_resolution=sphere_resolution,
            )
            if render_style == "wireframe":
                ax.plot_wireframe(x, y, z, color="tab:blue", linewidth=0.6, alpha=0.6)
            else:
                ax.plot_surface(x, y, z, color="tab:blue", linewidth=0.0, alpha=0.35, shade=True)

        if coordinate == "Cartesian":
            x1, x2 = design_region[0]
            y1, y2 = design_region[1]
            z1, z2 = design_region[2]
            corners = np.array(
                [
                    [x1, y1, z1],
                    [x1, y1, z2],
                    [x1, y2, z1],
                    [x1, y2, z2],
                    [x2, y1, z1],
                    [x2, y1, z2],
                    [x2, y2, z1],
                    [x2, y2, z2],
                ]
            )
            edges = [
                (0, 1),
                (0, 2),
                (0, 4),
                (1, 3),
                (1, 5),
                (2, 3),
                (2, 6),
                (3, 7),
                (4, 5),
                (4, 6),
                (5, 7),
                (6, 7),
            ]
            for i, j in edges:
                ax.plot(
                    [corners[i, 0], corners[j, 0]],
                    [corners[i, 1], corners[j, 1]],
                    [corners[i, 2], corners[j, 2]],
                    color="black",
                    linewidth=0.8,
                )
        elif coordinate == "Polar":
            r1, r2 = design_region[0]
            theta1, theta2 = design_region[1]
            z1, z2 = design_region[2]
            theta_values = np.linspace(theta1, theta2, 120)
            for radius in (r1, r2):
                xs = radius * np.cos(theta_values)
                ys = radius * np.sin(theta_values)
                ax.plot(xs, ys, np.full_like(theta_values, z1), color="black", linewidth=0.8)
                ax.plot(xs, ys, np.full_like(theta_values, z2), color="black", linewidth=0.8)
            for theta in (theta1, theta2):
                x_line = np.array([r1 * np.cos(theta), r2 * np.cos(theta)])
                y_line = np.array([r1 * np.sin(theta), r2 * np.sin(theta)])
                ax.plot(x_line, y_line, np.array([z1, z1]), color="black", linewidth=0.8)
                ax.plot(x_line, y_line, np.array([z2, z2]), color="black", linewidth=0.8)
                ax.plot(
                    np.full(2, r1 * np.cos(theta)),
                    np.full(2, r1 * np.sin(theta)),
                    np.array([z1, z2]),
                    color="black",
                    linewidth=0.8,
                )
                ax.plot(
                    np.full(2, r2 * np.cos(theta)),
                    np.full(2, r2 * np.sin(theta)),
                    np.array([z1, z2]),
                    color="black",
                    linewidth=0.8,
                )
        else:
            msg = f"Unsupported coordinate '{coordinate}' for 3D visualization."
            raise NGnetValidationError(msg)

        if no_axis:
            ax.set_axis_off()
        else:
            ax.set_title("Gaussian bases (3D)")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
        ax.set_box_aspect((1, 1, 1))
        _finalize_basis_plot(fig, filepath, check_basis_GUI=check_basis_GUI)

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
    coordinate: CoordinateSystem,
    sigma: float,
    design_region: tuple,
    distance_factor: float = 0.8,
    *,
    dimension: Dimension = "2D",
    eliminate_bases_on_edge: bool = False,
    normalize_output: bool = True,
    check_basis_GUI: bool = False,
    fixed_design_region: tuple | list | None = None,
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
        dimension (str): Spatial dimension. "2D" or "3D". Defaults to "2D".
        eliminate_bases_on_edge (bool, optional): Whether to remove basis functions located on the region boundary.
            Defaults to False.
        normalize_output (bool): If False, normalization of function output will be disabled.
            (so it will be no longer "NGnet", but just weighted sum of Gaussian bases). Defaults to True.
        check_basis_GUI (bool): If True, opens GUI for basis check; otherwise saves image.
        fixed_design_region (PhysicalRegion, optional): Variable region whose outside is fixed to level-set value 1.0.
    """
    ngnet = NGnet(normalize_output=normalize_output, dimension=dimension, coordinate=coordinate)
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
    elif dimension == "3D":
        _design_region = [list(axis) for axis in _to_3d_region(_design_region)]  # type: ignore[arg-type]
    else:
        msg = f"Unsupported dimension '{dimension}'."
        raise NGnetValidationError(msg)
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
    elif dimension == "3D":
        ngnet.visualize_gaussian_bases_radius_3d(
            _design_region,
            coordinate,
            "gaussian_3d.png",
            check_basis_GUI=check_basis_GUI,
            render_style="wireframe",
        )
    return ngnet
