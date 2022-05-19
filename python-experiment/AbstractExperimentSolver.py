import pandas as pd
import time as time

class AbstractExperimentSolver:

	def __init__(self) -> None:
		self.numberOfTimesToRun = 5
	
	def getInputMatrixAndOutputVector(self, fileName, separator=','):
		directory = './data/' + fileName
		df = pd.read_csv(filepath_or_buffer=directory, sep=separator)
		x = df.iloc[:,0:df.shape[1] - 1]
		y = df.iloc[:,df.shape[1] - 1]
		return x, y

	def getSmallExperimentData(self):
		return self.getInputMatrixAndOutputVector('feynman_I_10_7.csv')

	def getMediumExperimentData(self):
		return self.getInputMatrixAndOutputVector('1191_BNG_pbc.tsv', '\t')

	def runExperimentNTimes(self, x, y):
		# Measuring time
		startTime = time.time()
		for i in range(self.numberOfTimesToRun):
			reg = self.runRegressionSolver(x, y)

		return (time.time() - startTime) / self.numberOfTimesToRun

	def runSmallExperiment(self):
		x, y = self.getSmallExperimentData()
		return self.runExperimentNTimes(x, y)

	def runMediumExperiment(self):
		x, y = self.getMediumExperimentData()
		return self.runExperimentNTimes(x, y)
	
	def runRegressionSolver(self, x, y):
		pass