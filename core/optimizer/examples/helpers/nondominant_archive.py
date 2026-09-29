"""
nondominant_archive.py
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

from emsopt_engine.individual import Individual, Population


class NonDominatedArchive:
    """Minimization, strict Pareto dominance. Optional max_size with crowding-based truncation."""

    def __init__(self, n_obj: int, eps: float = 1e-6) -> None:
        self.archive = Population()
        self.F = np.empty((0, n_obj), dtype=float)
        self.CV = np.empty((0,), dtype=float)
        self.eps = eps

    @staticmethod
    def _to_array(ind: Individual) -> tuple[np.ndarray, float]:
        f = np.asarray(ind.metrics.objectives, dtype=float)
        cv = float(ind.metrics.constraint_violation)
        return f, cv

    def _epsilon_filter(self, f_new: np.ndarray) -> bool:
        if self.eps is None or self.F.size == 0:
            return False
        return np.any(np.max(np.abs(self.F - f_new), axis=1) <= self.eps)

    def add(self, ind: Individual) -> None:
        f_new, cv_new = self._to_array(ind)
        # avoid to add almost the same solution
        if self._epsilon_filter(f_new):
            return

        # add if archive is empty
        if self.F.size == 0:
            self.archive.add_individual(ind)
            self.F = f_new[None, :]
            self.CV = np.array([cv_new], dtype=float)
            return

        # if dominated, do not add
        feas_exist = self.CV == 0
        feas_new = cv_new == 0
        if not feas_new:
            dominated_by_exist = (feas_exist) | ((~feas_exist) & (cv_new > self.CV))
        else:
            le = (self.F[feas_exist] <= f_new).all(axis=1)
            lt = (self.F[feas_exist] < f_new).any(axis=1)
            dominated_by_exist = np.zeros_like(self.CV, dtype=bool)
            dominated_by_exist[np.where(feas_exist)[0]] = le & lt

        if np.any(dominated_by_exist):
            return

        # remove dominated ones by new ind
        if not feas_new:
            dominated_exist = (~feas_exist) & (cv_new < self.CV)
        else:
            ge = (self.F[feas_exist] >= f_new).all(axis=1)
            gt = (self.F[feas_exist] > f_new).any(axis=1)
            dominated_exist = np.zeros_like(self.CV, dtype=bool)
            dominated_exist[np.where(feas_exist)[0]] = ge & gt
            dominated_exist |= ~feas_exist

        keep = ~dominated_exist
        if not np.all(keep):
            self.F = self.F[keep]
            self.CV = self.CV[keep]
            self.archive = Population(
                {idx: ind for idx, (ind, k) in enumerate(zip(self.archive.values(), keep, strict=True)) if k}
            )

        # add
        self.archive.add_individual(ind)
        self.F = np.vstack([self.F, f_new[None, :]])
        self.CV = np.concatenate([self.CV, [cv_new]])

    def add_population(self, pop: Population) -> None:
        for ind in pop.values():
            self.add(ind)
        self.archive.reindex()
