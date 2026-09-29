import numpy as np

from emsopt_engine.individual import Population
from emsopt_engine.interface.evaluator_interface import EvaluatorInterface
from emsopt_engine.registry import evaluator


class DTLZ1(EvaluatorInterface):
    """
    Evaluation class for the DTLZ1 multi-objective benchmark function.
    Implements EvaluatorInterface.
    """

    def __init__(self, dim: int, num_obj: int) -> None:
        self.dim = dim
        self.num_obj = num_obj

    def evaluate(self, population: Population) -> Population:
        """
        Evaluates each individual's solution in the Population using DTLZ1 and sets the objectives.
        Args:
            population (Population): Population to be evaluated
        Returns:
            Population: Population with fitness values set
        """
        for ind in population.values():
            if len(ind.solution) != self.dim:
                msg = "Individual's solution dim doesn't match evaluator's dim."
                raise ValueError(msg)
            ind.metrics.objectives = self._dtlz1(ind.solution, self.num_obj)
        return population

    def evaluate_parallel(self, population: Population, num_processes: int | None) -> Population:
        # No need for parallelization, so call evaluate
        return self.evaluate(population)

    @staticmethod
    def _dtlz1(x: list[float], num_obj: int) -> list[float]:
        """
        Calculation of the DTLZ1 function.
        Args:
            x (list or np.ndarray): Solution vector
            n_obj (int): Number of objectives
        Returns:
            list: List of objective function values
        """
        x = np.asarray(x)
        n_var = x.size
        k = n_var - num_obj + 1
        g = 100 * (k + np.sum((x[-k:] - 0.5) ** 2 - np.cos(20 * np.pi * (x[-k:] - 0.5))))
        f = []
        for i in range(num_obj):
            prod = 0.5 * (1 + g)
            for j in range(num_obj - i - 1):
                prod *= x[j]
            if i != 0:
                prod *= 1 - x[num_obj - i - 1]
            f.append(prod)
        return f

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
        return self.num_obj


@evaluator("dtlz1")
def build_dtlz1(dim: int, num_obj: int) -> EvaluatorInterface:
    return DTLZ1(dim, num_obj)
