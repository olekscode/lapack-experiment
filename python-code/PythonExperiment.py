from ScikitLearnGradientDescentSolver import ScikitLearnGradientDescentSolver
from PythonGradientDescentSolver import PythonGradientDescentSolver
import csv

scikit_learn_gradient_descent_solver = ScikitLearnGradientDescentSolver()
scikit_learn_gradient_descent_small_time = scikit_learn_gradient_descent_solver.run_small_experiment()
scikit_learn_gradient_descent_medium_time = scikit_learn_gradient_descent_solver.run_medium_experiment()
scikit_learn_gradient_descent_big_time = scikit_learn_gradient_descent_solver.run_big_experiment()

python_gradient_descent_solver = PythonGradientDescentSolver()
python_gradient_descent_small_time = python_gradient_descent_solver.run_small_experiment()
python_gradient_descent_medium_time = python_gradient_descent_solver.run_medium_experiment()
python_gradient_descent_big_time = python_gradient_descent_solver.run_big_experiment()

fields = ['ScikitLearn Gradient Descent', 'Pure Python Gradient Descent']

# data rows of csv file 
rows = [[scikit_learn_gradient_descent_small_time, python_gradient_descent_small_time],
        [scikit_learn_gradient_descent_medium_time, python_gradient_descent_medium_time],
        [scikit_learn_gradient_descent_big_time, python_gradient_descent_big_time]]

# name of csv file 
filename = "./experiment-results/python-results.csv"

# writing to csv file 
with open(filename, 'w') as csvfile: 
    # creating a csv writer object 
    csv_writer = csv.writer(csvfile)
    
    # writing the fields 
    csv_writer.writerow(fields)
    
    # writing the data rows 
    csv_writer.writerows(rows)