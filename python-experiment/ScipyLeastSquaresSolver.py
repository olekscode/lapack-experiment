from scipy.linalg import lstsq
from AbstractExperimentSolver import AbstractExperimentSolver

class ScipyLeastSquaresSolver(AbstractExperimentSolver):

	def runRegressionSolver(self, x, y):
		return lstsq(x, y)