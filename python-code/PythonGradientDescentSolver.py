from AbstractExperimentSolver import AbstractExperimentSolver
from PythonGradientDescent import PythonGradientDescent


class PythonGradientDescentSolver(AbstractExperimentSolver):

    def __init__(self):
        super().__init__()
        self.n = 1

    def run_regression_solver(self, x, y):
        return PythonGradientDescent().fit(x.values, y.values)
