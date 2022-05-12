import pandas as pd

class PythonExperimentSetUp:

	def __init__(self) -> None:
		self.numberOfTimesToRun = 5
	
	def getInputMatrixAndOutputVector(self, fileName, separator=','):
		directory = '../data/' + fileName
		df = pd.read_csv(directory, separator)
		x = df.iloc[:,0:df.shape[1] - 1]
		y = df.iloc[:,df.shape[1] - 1]
		return x, y

	def getSmallExperimentData(self):
		return self.getInputMatrixAndOutputVector('feynman_I_10_7.csv')

	def getMediumExperimentData(self):
		return self.getInputMatrixAndOutputVector('1191_BNG_pbc.tsv', '\t')