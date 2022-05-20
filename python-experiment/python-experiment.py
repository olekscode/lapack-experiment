from ScikitLearnGradientDescentSolver import ScikitLearnGradientDescentSolver
from ScipyLeastSquaresSolver import ScipyLeastSquaresSolver
import csv

scipyLeastSquaresSolver = ScipyLeastSquaresSolver()
scipyLeastSquaresSmallTime = scipyLeastSquaresSolver.runSmallExperiment()
scipyLeastSquaresMediumTime = scipyLeastSquaresSolver.runMediumExperiment()

scikitLearnGradientDescentSolver = ScikitLearnGradientDescentSolver()
scikitLearnGradientDescentSmallTime = scikitLearnGradientDescentSolver.runSmallExperiment()
scikitLearnGradientDescentMediumTime = scikitLearnGradientDescentSolver.runMediumExperiment()

# purePythonGradientDescentSmallTime = 0
# purePythonGradientDescentMediumTime = 0

fields = ['SciPy LeastSquares', 'ScikitLearn Gradient Descent', 'Pure Python Gradient Descent'] 

# data rows of csv file 
rows = [ [scipyLeastSquaresSmallTime, scikitLearnGradientDescentSmallTime, '-'],
         [scipyLeastSquaresMediumTime, scikitLearnGradientDescentMediumTime, '-'] ] 

# name of csv file 
filename = "./experiment-results/python-results.csv"

# writing to csv file 
with open(filename, 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
    
    # writing the fields 
    csvwriter.writerow(fields) 
    
    # writing the data rows 
    csvwriter.writerows(rows)