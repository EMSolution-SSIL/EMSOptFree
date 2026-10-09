"""
zdt1.py
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

from emsopt_engine.individual import Population
from emsopt_engine.interface.evaluator_interface import EvaluatorInterface
from emsopt_engine.registry import evaluator


class ZDT1(EvaluatorInterface):
    """
    Evaluator for ZDT1 benchmark function (2 objectives)
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def evaluate(self, population: Population) -> Population:
        for ind in population.values():
            if len(ind.solution) != self.dim:
                msg = "Individual's solution dim doesn't match evaluator's dim."
                raise ValueError(msg)
            x = np.array(ind.solution)
            f1 = x[0]
            g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
            f2 = g * (1 - np.sqrt(f1 / g))
            ind.metrics.objectives = [f1, f2]
        return population

    def evaluate_parallel(self, population: Population, num_processes: int | None = None) -> Population:  # noqa: ARG002
        # Simple implementation: just call evaluate (no real parallelism)
        return self.evaluate(population)

    def get_variable_dimension(self) -> int:
        """Get variable dimension

        Returns:
            int: dimension
        """
        return self.dim

    def get_num_objectives(self) -> int:
        """Get number of objectives

        Returns:
            int: number of objectives
        """
        return 2


@evaluator("zdt1")
def build_zdt1(dim: int) -> EvaluatorInterface:
    return ZDT1(dim)
