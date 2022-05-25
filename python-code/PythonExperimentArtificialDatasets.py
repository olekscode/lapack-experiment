from ScikitLearnLeastSquaresSolver import ScikitLearnLeastSquaresSolver
from PythonGradientDescentSolver import PythonGradientDescentSolver
from ScikitLearnGradientDescentSolver import ScikitLearnGradientDescentSolver
import csv


def run_small(solver, solver_name):
    time = solver.run_small_experiment()
    print('Finished ', solver_name, ' with small dataset in ', time, ' seconds')
    return time


def run_medium(solver, solver_name):
    time = solver.run_medium_experiment()
    print('Finished ', solver_name, ' with medium dataset in ', time, ' seconds')
    return time


def run_big(solver, solver_name):
    time = solver.run_big_experiment()
    print('Finished ', solver_name, ' with big dataset in ', time, ' seconds')
    return time


# Scikit Learn least squares
scikit_least_squares_solver = ScikitLearnLeastSquaresSolver()
scikit_least_squares_solver_name = 'ScikitLearn Least Squares'

scikit_least_squares_small = run_small(scikit_least_squares_solver, scikit_least_squares_solver_name)
scikit_least_squares_medium = run_medium(scikit_least_squares_solver, scikit_least_squares_solver_name)
scikit_least_squares_big = run_big(scikit_least_squares_solver, scikit_least_squares_solver_name)


# Python gradient descent
python_gradient_descent_solver = PythonGradientDescentSolver()
python_gradient_descent_solver_name = 'Python Gradient Descent'

python_gradient_descent_small = run_small(python_gradient_descent_solver, python_gradient_descent_solver_name)
python_gradient_descent_medium = run_medium(python_gradient_descent_solver, python_gradient_descent_solver_name)
# python_gradient_descent_big = run_big(python_gradient_descent_solver, python_gradient_descent_solver_name)


# Scikit Learn gradient descent
scikit_gradient_descent_solver = ScikitLearnGradientDescentSolver()
scikit_gradient_descent_solver_name = 'ScikitLearn Gradient Descent'

scikit_gradient_descent_small = run_small(scikit_gradient_descent_solver, scikit_gradient_descent_solver_name)
scikit_gradient_descent_medium = run_medium(scikit_gradient_descent_solver, scikit_gradient_descent_solver_name)
scikit_gradient_descent_big = run_big(scikit_gradient_descent_solver, scikit_gradient_descent_solver_name)


# Writing the csv file
fields = ['ScikitLearn Least Squares', 'Pure Python Gradient Descent', 'ScikitLearn Gradient Descent']

rows = [[scikit_least_squares_small, python_gradient_descent_small, scikit_gradient_descent_small],
        [scikit_least_squares_medium, python_gradient_descent_medium, scikit_gradient_descent_medium],
        [scikit_least_squares_big, '-', scikit_gradient_descent_big]]

filename = "../experiment-results/python-results-artificial-datasets.csv"

with open(filename, 'w') as csvfile:
    csv_writer = csv.writer(csvfile)

    csv_writer.writerow(fields)
    csv_writer.writerows(rows)
    print('Generated csv file named: ', filename)
