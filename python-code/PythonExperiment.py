from ScikitLearnLeastSquaresSolver import ScikitLearnLeastSquaresSolver
from PythonGradientDescentSolver import PythonGradientDescentSolver
from ScikitLearnGradientDescentSolver import ScikitLearnGradientDescentSolver
import csv

print('Starting experiment')

scikit_learn_least_squares_solver = ScikitLearnLeastSquaresSolver()
scikit_learn_least_squares_small = scikit_learn_least_squares_solver.run_small_experiment()
print('Finished ScikitLearn Least Squares with small dataset')
scikit_learn_least_squares_medium = scikit_learn_least_squares_solver.run_medium_experiment()
print('Finished ScikitLearn Least Squares with medium dataset')
scikit_learn_least_squares_big = scikit_learn_least_squares_solver.run_big_experiment()
print('Finished ScikitLearn Least Squares with big dataset')

python_gradient_descent_solver = PythonGradientDescentSolver()
python_gradient_descent_small = python_gradient_descent_solver.run_small_experiment()
print('Finished Python Gradient Descent with small dataset')
python_gradient_descent_medium = python_gradient_descent_solver.run_medium_experiment()
print('Finished Python Gradient Descent with medium dataset')
# python_gradient_descent_big = python_gradient_descent_solver.run_big_experiment()
# print('Finished Python Gradient Descent with big dataset')

scikit_learn_gradient_descent_solver = ScikitLearnGradientDescentSolver()
scikit_learn_gradient_descent_small = scikit_learn_gradient_descent_solver.run_small_experiment()
print('Finished ScikitLearn Gradient Descent with small dataset')
scikit_learn_gradient_descent_medium = scikit_learn_gradient_descent_solver.run_medium_experiment()
print('Finished ScikitLearn Gradient Descent with medium dataset')
scikit_learn_gradient_descent_big = scikit_learn_gradient_descent_solver.run_big_experiment()
print('Finished ScikitLearn Gradient Descent with big dataset')

fields = ['ScikitLearn Least Squares', 'Pure Python Gradient Descent', 'ScikitLearn Gradient Descent']

rows = [[scikit_learn_least_squares_small, python_gradient_descent_small, scikit_learn_gradient_descent_small],
        [scikit_learn_least_squares_medium, python_gradient_descent_medium, scikit_learn_gradient_descent_medium],
        [scikit_learn_least_squares_big, '-', scikit_learn_gradient_descent_big]]

filename = "../experiment-results/python-results.csv"

with open(filename, 'w') as csvfile:
    csv_writer = csv.writer(csvfile)

    csv_writer.writerow(fields)
    csv_writer.writerows(rows)
