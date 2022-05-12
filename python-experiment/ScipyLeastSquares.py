from scipy.linalg import lstsq
import ExperimentSetUp
import time

class ScipyLeastSquares:

	def __init__(self) -> None:
		self.numberOfTimesToRun = ExperimentSetUp.PythonExperimentSetUp().numberOfTimesToRun

	def runSmallExperiment(self):
		x, y = ExperimentSetUp.PythonExperimentSetUp().getSmallExperimentData()

		# Measuring time
		startTime = time.time()
		for i in range(self.numberOfTimesToRun):
			coef, residues, rank, singular = lstsq(x, y)

		return (time.time() - startTime) / self.numberOfTimesToRun

	def runMediumExperiment(self):
		x, y = ExperimentSetUp.PythonExperimentSetUp().getMediumExperimentData()

		# Measuring time
		startTime = time.time()
		for i in range(self.numberOfTimesToRun):
			coef, residues, rank, singular = lstsq(x, y)

		return (time.time() - startTime) / self.numberOfTimesToRun