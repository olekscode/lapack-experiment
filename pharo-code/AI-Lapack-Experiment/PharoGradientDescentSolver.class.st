"
I solve the linear regression using the gradient descent algorithm
"
Class {
	#name : #PharoGradientDescentSolver,
	#superclass : #AbstractLinearRegressionSolver,
	#category : #'AI-Lapack-Experiment'
}

{ #category : #'experiment definition' }
PharoGradientDescentSolver class >> linearRegressionSolvingAlgorithmClass [

	^ AILinearRegression
]
