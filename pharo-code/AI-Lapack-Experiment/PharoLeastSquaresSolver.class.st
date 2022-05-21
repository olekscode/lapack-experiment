"
I solve the linear regression using the least squares algorithm implemented in Pharo.
"
Class {
	#name : #PharoLeastSquaresSolver,
	#superclass : #AbstractLinearRegressionSolver,
	#category : #'AI-Lapack-Experiment'
}

{ #category : #'experiment definition' }
PharoLeastSquaresSolver class >> linearRegressionSolvingAlgorithmClass [

	^ AILinearRegressionLeastSquaresVanilla
]
