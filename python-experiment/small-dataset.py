from sklearn.linear_model import LinearRegression
from scipy.linalg import lstsq
import pandas as pd
import numpy as np
import time

df = pd.read_csv('./feynman_I_10_7.csv')

x = df.iloc[:,0:df.shape[1] - 1]
y = df.iloc[:,df.shape[1] - 1]

startTime = time.time()

#####your python script#####
for i in range(5):
	coef, residues, rank, singular = lstsq(x, y)

######
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime / 5))