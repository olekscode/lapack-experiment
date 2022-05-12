from sklearn.linear_model import LinearRegression
import AbstractExperimentSolver
import time

class ScikitLearnGradientDescent(AbstractExperimentSolver.AbstractExperimentSolver):

	def runRegressionSolver(self, x, y):
		return LinearRegression().fit(x, y)
	
