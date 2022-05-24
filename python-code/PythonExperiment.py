from ScikitLearnGradientDescentSolver import ScikitLearnGradientDescentSolver
from PythonGradientDescentSolver import PythonGradientDescentSolver
import csv

print('Starting experiment')

scikit_learn_gradient_descent_solver = ScikitLearnGradientDescentSolver()
scikit_learn_gradient_descent_small_time = scikit_learn_gradient_descent_solver.run_small_experiment()
print('Finished ScikitLearn with small dataset')
scikit_learn_gradient_descent_medium_time = scikit_learn_gradient_descent_solver.run_medium_experiment()
print('Finished ScikitLearn with medium dataset')
scikit_learn_gradient_descent_big_time = scikit_learn_gradient_descent_solver.run_big_experiment()
print('Finished ScikitLearn with big dataset')

python_gradient_descent_solver = PythonGradientDescentSolver()
python_gradient_descent_small_time = python_gradient_descent_solver.run_small_experiment()
print('Finished Python Gradient Descent with small dataset')
python_gradient_descent_medium_time = python_gradient_descent_solver.run_medium_experiment()
print('Finished Python Gradient Descent with medium dataset')
# python_gradient_descent_big_time = python_gradient_descent_solver.run_big_experiment()
# print('Finished Python Gradient Descent with big dataset')

fields = ['ScikitLearn Least Squares', 'Pure Python Gradient Descent']

rows = [[scikit_learn_gradient_descent_small_time, python_gradient_descent_small_time],
        [scikit_learn_gradient_descent_medium_time, python_gradient_descent_medium_time],
        [scikit_learn_gradient_descent_big_time, '-']]

filename = "../experiment-results/python-results.csv"

with open(filename, 'w') as csvfile:
    csv_writer = csv.writer(csvfile)

    csv_writer.writerow(fields)
    csv_writer.writerows(rows)
