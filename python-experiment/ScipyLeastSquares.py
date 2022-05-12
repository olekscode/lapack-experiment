from scipy.linalg import lstsq
import AbstractExperimentSolver

class ScipyLeastSquares(AbstractExperimentSolver.AbstractExperimentSolver):

	def runRegressionSolver(self, x, y):
		return lstsq(x, y)