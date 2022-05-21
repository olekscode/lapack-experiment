from sklearn.linear_model import LinearRegression
from AbstractExperimentSolver import AbstractExperimentSolver


class ScikitLearnGradientDescentSolver(AbstractExperimentSolver):

	def run_regression_solver(self, x, y):
		return LinearRegression().fit(x, y)
