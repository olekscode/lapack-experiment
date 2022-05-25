import pandas
import time as time


class AbstractExperimentSolver:

	def __init__(self):
		self.n = 5

	def get_input_matrix_and_output_vector(self, file_name, separator=','):
		directory = '../data/' + file_name
		df = pandas.read_csv(filepath_or_buffer=directory, sep=separator)  # header=None
		x = df.iloc[:, 0:df.shape[1] - 1]
		y = df.iloc[:, df.shape[1] - 1]
		return x, y

	def get_small_experiment_data(self):
		return self.get_input_matrix_and_output_vector('feynman_I_10_7.tsv', '\t')  # 'small_dataset.csv'

	def get_medium_experiment_data(self):
		return self.get_input_matrix_and_output_vector('1191_BNG_pbc.tsv', '\t')  # 'medium_dataset.csv'

	def get_big_experiment_data(self):
		return self.get_input_matrix_and_output_vector('big_dataset.csv')

	def experiment(self, x, y):
		start_time = time.time()
		for i in range(self.n):
			reg = self.run_regression_solver(x, y)

		return (time.time() - start_time) / self.n

	def run_experiment_n_times(self, x, y):
		try:
			return self.experiment(x, y)
		except BaseException as exception:
			print(exception)
			return str(exception).replace(',', '')

	def run_small_experiment(self):
		x, y = self.get_small_experiment_data()
		return self.run_experiment_n_times(x, y)

	def run_medium_experiment(self):
		x, y = self.get_medium_experiment_data()
		return self.run_experiment_n_times(x, y)

	def run_big_experiment(self):
		x, y = self.get_big_experiment_data()
		return self.run_experiment_n_times(x, y)
	
	def run_regression_solver(self, x, y):
		pass
