import pandas as pd
import time as time


class AbstractExperimentSolver:
	
	def get_input_matrix_and_output_vector(self, file_name, separator=','):
		directory = '../data/' + file_name
		df = pd.read_csv(filepath_or_buffer=directory, sep=separator)
		x = df.iloc[:,0:df.shape[1] - 1]
		y = df.iloc[:,df.shape[1] - 1]
		return x, y

	def get_small_experiment_data(self):
		return self.get_input_matrix_and_output_vector('feynman_I_10_7.csv')

	def get_medium_experiment_data(self):
		return self.get_input_matrix_and_output_vector('1191_BNG_pbc.tsv', '\t')

	def get_big_experiment_data(self):
		return self.get_input_matrix_and_output_vector('huge_dataset.csv')

	def run_experiment_n_times(self, x, y, n=5):
		# Measuring time
		start_time = time.time()
		for i in range(n):
			reg = self.run_regression_solver(x, y)

		return (time.time() - start_time) / n

	def run_small_experiment(self):
		x, y = self.get_small_experiment_data()
		return self.run_experiment_n_times(x, y)

	def run_medium_experiment(self):
		x, y = self.get_medium_experiment_data()
		return self.run_experiment_n_times(x, y)

	def run_big_experiment(self):
		x, y = self.get_big_experiment_data()
		return self.run_experiment_n_times(x, y, 1)
	
	def run_regression_solver(self, x, y):
		pass
