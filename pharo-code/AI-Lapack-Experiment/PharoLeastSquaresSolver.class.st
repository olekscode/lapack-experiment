"
I solve the linear regression using the least squares algorithm implemented in Pharo.
"
Class {
	#name : #PharoLeastSquaresSolver,
	#superclass : #AbstractLinearRegressionSolver,
	#category : #'AI-Lapack-Experiment'
}

{ #category : #initialization }
PharoLeastSquaresSolver >> initialize [

	super initialize.
	n := 1
]

{ #category : #private }
PharoLeastSquaresSolver >> runRegressionSolverX: x y: y [

	^ AILinearRegressionLeastSquaresVanilla new fitX: x y: y
]
