"""
decomposition.py
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

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations_with_replacement

import numpy as np

from emsopt_engine.individual import Population

ArrayLike = Iterable[float]
EPS = 1e-10
DEFAULT_MAX_VALUE = 1e9
DEFAULT_MIN_NADIR_DIFF = 1e-12


def to_1d(a: Iterable[float]) -> np.ndarray:
    """convert arraylike object to 1D np.ndarray

    Args:
        a (Iterable[float]): arraylike object

    Returns:
        np.ndarray: converted a
    """
    return np.asarray(a, dtype=float).ravel()


def check_shapes(a1: np.ndarray, a2: np.ndarray) -> None:
    """Raise error when a1.shape != a2.shape

    Args:
        a1 (np.ndarray): array
        a2 (np.ndarray): array

    Raises:
        ValueError: mismatch shape
    """
    if a1.shape != a2.shape:
        msg = f"Shape mismatch: f{a1.shape} vs lam{a2.shape}"
        raise ValueError(msg)


def generate_weights_grid(num_objectives: int, divisions: int) -> np.ndarray:
    """
    Generate uniformly distributed weight vectors on a simplex
    using the 'grid' (stars-and-bars) method for MOEA/D.

    Args:
        num_objectives (int): Number of objectives (m).
        divisions (int): Number of divisions (H) of the unit interval in each objective.
            Weight components take values in {0, 1/H, 2/H, ..., 1}.

    Returns:
        np.ndarray of shape (num_weights, num_objectives). Each row is a weight vector λ with λ_i >= 0 and sum λ_i = 1.

    Raises:
        ValueError: if num_objectives < 2 or divisions < 1
    """
    if num_objectives < 2:  # noqa: PLR2004
        msg = "num_objectives must be >= 2"
        raise ValueError(msg)
    if divisions < 1:
        msg = "divisions must be >= 1"
        raise ValueError(msg)

    weights_list: list[np.ndarray] = []

    # cut points in [0, H], pick m-1 of them (sorted, unique)
    for cuts in combinations_with_replacement(range(divisions + 1), num_objectives - 1):
        # prepend 0 and append H to make m segments
        pts = (0, *cuts, divisions)
        # differences are nonnegative integers summing to H
        counts = np.diff(pts)  # shape (m,)
        w = counts.astype(float) / divisions  # normalize to sum 1
        weights_list.append(w)
    return np.vstack(weights_list)


def generate_weights_dirichlet(num_objectives: int, num_samples: int, seed: int | None = None) -> np.ndarray:
    """
    High-dimensional fallback: sample n points on the simplex via Dirichlet(1,...,1).
    Sums to 1 by construction; coverage is random (not a perfect grid).

    Args:
        num_objectives (int): Number of objectives (m).
        num_samples (int): Number of weight vectors to generate (n).
        seed (int | None): Random seed for reproducibility.

    Returns:
        np.ndarray of shape (num_samples, num_objectives). Each row is a weight vector λ with λ_i >= 0 and sum λ_i = 1.

    Raises:
        ValueError: if num_objectives < 2 or num_samples < 1
    """
    if num_objectives < 2:  # noqa: PLR2004
        msg = "num_objectives must be >= 2"
        raise ValueError(msg)
    if num_samples < 1:
        msg = "num_samples must be >= 1"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    return rng.dirichlet(alpha=np.ones(num_objectives), size=num_samples)


def neighbor_indices(w: np.ndarray, t: int) -> np.ndarray:
    """
    Compute T-nearest neighbors in weight space (Euclidean) for each weight vector.
    Returns shape (len(W), T) with indices (including self at column 0).

    Args:
        w (np.ndarray): Weight vectors of shape (N, M).
        t (int): Number of neighbors to find (T).

    Returns:
        np.ndarray of shape (N, T) with neighbor indices for each weight vector.

    Raises:
        ValueError: if t < 1
    """
    if t < 1:
        msg = "T >= 1 is expected"
        raise ValueError(msg)
    # pairwise distances
    # (w[:, None, :] - w[None, :, :]) : (N, N, M) pairwise diff vector matrix
    # d : (N, N) pairwise distance matrix
    # For moderate sizes; for huge N, use faiss/annoy/scipy KDTree as needed
    d = np.sum((w[:, None, :] - w[None, :, :]) ** 2, axis=2)
    return np.argsort(d, axis=1)[:, :t]


def initialize_reference_and_nadir(
    num_objectives: int,
    reference: ArrayLike | None = None,
    nadir: ArrayLike | None = None,
) -> tuple[list[float], list[float]]:
    """Initialize reference and nadir points for decomposition-based optimization."""
    z_ref = list(reference) if reference is not None else [float("inf")] * num_objectives
    z_nadir = list(nadir) if nadir is not None else [-float("inf")] * num_objectives
    return z_ref, z_nadir


def update_reference_and_nadir(
    archive: Population,
    num_objectives: int,
    scalarizer: "Scalarizer",
    reference: ArrayLike | None = None,
    nadir: ArrayLike | None = None,
    max_value: float = DEFAULT_MAX_VALUE,
    min_nadir_diff: float = DEFAULT_MIN_NADIR_DIFF,
) -> tuple[list[float], list[float]]:
    """Update reference/nadir points from archive and reflect them to the scalarizer."""
    z_ref, z_nadir = initialize_reference_and_nadir(num_objectives, reference, nadir)

    cur_min = [float("inf")] * num_objectives
    cur_max = [-float("inf")] * num_objectives
    for ind in archive.values():
        for i, objective in enumerate(ind.metrics.objectives):
            cur_min[i] = min(cur_min[i], objective)
            if objective < max_value:
                cur_max[i] = max(cur_max[i], objective)

    for i in range(num_objectives):
        if cur_min[i] < z_ref[i] - min_nadir_diff:
            z_ref[i] = cur_min[i]

    for i in range(num_objectives):
        if cur_max[i] > z_nadir[i] - min_nadir_diff:
            z_nadir[i] = cur_max[i]

    for i in range(num_objectives):
        if z_nadir[i] - z_ref[i] < min_nadir_diff:
            z_nadir[i] = z_ref[i] + min_nadir_diff

    scalarizer.reference = z_ref
    scalarizer.nadir = z_nadir
    return z_ref, z_nadir


"""
decomposition strategies
"""


@dataclass
class Scalarizer:
    """Base class (minimization)."""

    _reference: np.ndarray | None = None  # ideal point z*
    _nadir: np.ndarray | None = None  # nadir point (optional)
    use_normalization: bool = True  # normalize objectives before eval

    @property
    def reference(self) -> np.ndarray | None:
        return self._reference

    @reference.setter
    def reference(self, value: ArrayLike) -> None:
        self._reference = to_1d(value)

    @property
    def nadir(self) -> np.ndarray | None:
        return self._nadir

    @nadir.setter
    def nadir(self, value: ArrayLike) -> None:
        self._nadir = to_1d(value)

    @staticmethod
    def _normalize_objectives(objectives: np.ndarray, reference: np.ndarray, nadir: np.ndarray | None) -> np.ndarray:
        """
        Minimize-style normalization: (f - z*) / (z_nadir - z*).
        Falls back to (f - z*) if nadir is None. Clips tiny denominators.
        """
        if reference is None:
            return objectives
        objectives = objectives - reference
        if nadir is None:
            return objectives
        denom = nadir - reference
        denom = np.where(np.abs(denom) < EPS, 1.0, denom)  # avoid div by ~0
        return objectives / denom

    def _prep(self, objectves: ArrayLike, weights: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        objectves = to_1d(objectves)
        weights = to_1d(weights)
        check_shapes(objectves, weights)
        if self.use_normalization:
            _objectives = self._normalize_objectives(objectves, self._reference, self._nadir)
            reference_eff = np.zeros_like(_objectives)  # when normalized, reference point turns into origin
        else:
            _objectives = objectves.copy()
            reference_eff = self._reference if self._reference is not None else np.zeros_like(_objectives)
        return _objectives, weights, reference_eff

    def __call__(self, f: ArrayLike, lam: ArrayLike) -> float:
        raise NotImplementedError


class WeightedSum(Scalarizer):
    r"""g_ws(x|λ) = Σ_i λ_i f_i(x)  (minimization)"""

    def __init__(self, **kwargs: dict) -> None:
        super().__init__(**kwargs)
        self.use_normalization = False

    def __call__(self, f: ArrayLike, lam: ArrayLike) -> float:
        f, lam, _ = self._prep(f, lam)
        # weights should be nonnegative; normalize if they don't sum to 1
        s = lam.sum()
        w = lam / s if s > 0 else lam
        return float(np.dot(w, f))


class Tchebycheff(Scalarizer):
    r"""g_te(x|λ, z*) = max_i λ_i * | f_i(x) - z*_i |  (minimization)"""

    def __init__(self, **kwargs: dict) -> None:
        super().__init__(**kwargs)
        self.use_normalization = True

    def __call__(self, f: ArrayLike, lam: ArrayLike) -> float:
        f, lam, _ = self._prep(f, lam)  # f = (f - z*) / (z_nadir - z*)
        # normalize weights to avoid scaling side-effects (optional but common)
        w = lam.copy()
        w[w <= 0] = EPS
        return float(np.max(w * f))


class PBI(Scalarizer):
    r"""
    Penalty-based Boundary Intersection (minimization)
      g_pbi = d1 + θ * d2
    where d1 is the projection length of (f - z*) onto λ̂,
          d2 is the perpendicular distance to the λ̂-ray from z*.
    """

    def __init__(self, theta: float = 5.0, **kwargs: dict) -> None:
        super().__init__(**kwargs)
        self.theta = float(theta)
        self.use_normalization = False

    def __call__(self, f: ArrayLike, lam: ArrayLike) -> float:
        f, lam, z = self._prep(f, lam)
        # direction vector
        norm = np.linalg.norm(lam)
        if norm < EPS:
            msg = "Weight vector λ has near-zero norm."
            raise ValueError(msg)
        lam_hat = lam / norm

        v = f - z  # vector from reference point
        d1 = np.dot(v, lam_hat)  # signed length along lam_hat
        proj = z + d1 * lam_hat  # projection point on the ray
        d2 = np.linalg.norm(f - proj)  # perpendicular distance
        return float(d1 + self.theta * d2)


def generate_scalarizer(decomposition_type: str) -> Scalarizer:
    if decomposition_type.lower() == "pbi":
        return PBI()
    if decomposition_type.lower() == "weighted_sum":
        return WeightedSum()
    if decomposition_type.lower() == "tchebycheff":
        return Tchebycheff()
    msg = f"Unknown decomposition_type: {decomposition_type}"
    raise ValueError(msg)
