"""
optimization_analyzer.py
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

from logging import getLogger

import numpy as np
from sklearn.decomposition import PCA

from emsopt_engine.individual import Population
from utils.visualization import draw_2d_scatter

logger = getLogger(__name__)


class OptimizationAnalyzer:
    def __init__(self) -> None:
        self.pca: PCA | None = None
        self.pca_plot_lim: tuple[float, float] = ()

    def fit_pca(self, population: Population) -> None:
        """
        Fit PCA to the population solutions for visualization.
        Args:
            population (Population): Population to fit PCA on
        """
        self.pca = PCA(n_components=2)
        solutions = [ind.solution for ind in population.values()]
        arr = np.array(solutions)
        self.pca.fit(arr)
        lim = np.max(np.abs(arr))
        self.pca_plot_lim = (-1.0 * lim * 3.0, lim * 3.0)

    def plot_population(self, population: Population, filepath: str) -> None:
        """
        **fit_pca() must be called at least once before this method**
        Visualize the solution distribution of individuals in the population on 2D coordinates.
        for 3D or higher, maps to 2D using principal component analysis (PCA).

        Args:
            population: Population to visualize
            filepath (str, optional): Destination to save visualization result
        Returns:
            None
        Raises:
            ValueError: If PCA is not fitted before plotting 3D or higher dimensions
            ValueError: If invalid solution array dimension
            ValueError: If solution is 1D
        """
        # Extract solutions from population
        solutions = [ind.solution for ind in population.values()]
        arr = np.array(solutions)
        if arr.ndim != 2 or len(arr) < 2:  # noqa: PLR2004
            msg = "Invalid solution array dimension"
            raise ValueError(msg)
        n_dim = arr.shape[1]
        if n_dim == 1:
            msg = "Cannot plot 2D distribution for 1D solution"
            raise ValueError(msg)
        if n_dim == 2:  # noqa: PLR2004
            x, y = arr[:, 0], arr[:, 1]
        else:
            if self.pca is None:
                msg = "PCA model is not fitted. Call fit_pca() before plotting."
                raise ValueError(msg)
            arr_2d = self.pca.transform(arr)
            x, y = arr_2d[:, 0], arr_2d[:, 1]
        draw_2d_scatter(x, y, filepath, self.pca_plot_lim, self.pca_plot_lim)
