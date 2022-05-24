from AbstractExperimentSolver import AbstractExperimentSolver
from sklearn.linear_model import SGDRegressor


class ScikitLearnGradientDescentSolver(AbstractExperimentSolver):

    def run_regression_solver(self, x, y):
        return SGDRegressor().fit(x.values, y.values)
