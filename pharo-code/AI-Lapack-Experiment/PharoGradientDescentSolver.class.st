"
I solve the linear regression using the gradient descent algorithm
"
Class {
	#name : #PharoGradientDescentSolver,
	#superclass : #AbstractLinearRegressionSolver,
	#category : #'AI-Lapack-Experiment'
}

{ #category : #private }
PharoGradientDescentSolver >> runRegressionSolverX: x y: y [

	^ AILinearRegression new fitX: x y: y
]
