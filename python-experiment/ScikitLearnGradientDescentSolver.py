from sklearn.linear_model import LinearRegression
import AbstractExperimentSolver

class ScikitLearnGradientDescentSolver(AbstractExperimentSolver.AbstractExperimentSolver):

	def runRegressionSolver(self, x, y):
		return LinearRegression().fit(x, y)
	
