from AbstractExperimentSolver import AbstractExperimentSolver
from PythonGradientDescent import PythonGradientDescent


class PythonGradientDescentSolver(AbstractExperimentSolver):

    def run_regression_solver(self, x, y):
        return PythonGradientDescent().fit(x, y)
