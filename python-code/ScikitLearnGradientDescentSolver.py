from AbstractExperimentSolver import AbstractExperimentSolver
from sklearn.linear_model import SGDRegressor


class ScikitLearnGradientDescentSolver(AbstractExperimentSolver):

    def __init__(self):
        super().__init__()
        self.n = 1

    def run_regression_solver(self, x, y):
        return SGDRegressor().fit(x.values, y.values)
