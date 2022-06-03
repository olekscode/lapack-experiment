"
I solve the linear regression using the least squares algorithm implemented in Pharo.
"
Class {
	#name : #PharoLeastSquaresSolver,
	#superclass : #AbstractLinearRegressionSolver,
	#category : #'AI-Lapack-Experiment'
}

{ #category : #private }
PharoLeastSquaresSolver >> runRegressionSolverX: x y: y [

	^ AILinearRegressionLeastSquaresVanilla new fitX: x y: y
]
