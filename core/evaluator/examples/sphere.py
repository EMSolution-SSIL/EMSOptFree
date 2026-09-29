"""
sphere.py
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

from emsopt_engine.individual import Population
from emsopt_engine.interface.evaluator_interface import EvaluatorInterface
from emsopt_engine.registry import evaluator


class Sphere(EvaluatorInterface):
    """
    Sphere function class
    """

    def __init__(self, dim: int):
        self.dim = dim

    def evaluate(self, population: Population) -> Population:
        for ind in population.values():
            if len(ind.solution) != self.dim:
                msg = "Individual's solution dim doesn't match evaluator's dim."
                raise ValueError(msg)
            fitness = np.linalg.norm(np.array(ind.solution)) ** 2
            ind.metrics.objectives = [fitness]
        return population

    def evaluate_parallel(self, population: Population, num_processes: int | None = None) -> Population:
        """no parallelization but call self.evaluate"""
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
        return 1


@evaluator("sphere")
def build_sphere(dim: int) -> EvaluatorInterface:
    return Sphere(dim)
