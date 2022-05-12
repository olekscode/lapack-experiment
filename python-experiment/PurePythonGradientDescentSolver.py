from AbstractExperimentSolver import AbstractExperimentSolver
from PythonGradientDescent import PythonGradientDescent

class PurePythonGradientDescentSolver(AbstractExperimentSolver):

	def runRegressionSolver(self, x, y):
		return PythonGradientDescent().fit(x, y)