from sklearn.linear_model import LinearRegression
import ExperimentSetUp
import time

class ScikitLearnGradientDescent:

	def __init__(self) -> None:
		self.numberOfTimesToRun = ExperimentSetUp.PythonExperimentSetUp().numberOfTimesToRun
	
	def runSmallExperiment(self):
		x, y = ExperimentSetUp.PythonExperimentSetUp().getSmallExperimentData()

		# Measuring time
		startTime = time.time()
		for i in range(self.numberOfTimesToRun):
			reg = LinearRegression().fit(x, y)

		return (time.time() - startTime) / self.numberOfTimesToRun

	def runMediumExperiment(self):
		x, y = ExperimentSetUp.PythonExperimentSetUp().getMediumExperimentData()

		# Measuring time
		startTime = time.time()
		for i in range(self.numberOfTimesToRun):
			reg = LinearRegression().fit(x, y)

		return (time.time() - startTime) / self.numberOfTimesToRun
	
