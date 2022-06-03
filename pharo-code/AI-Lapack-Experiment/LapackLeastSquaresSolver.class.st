"
I solve the linear regression using the least squares algorithm calling LAPACK Fortran library for speed up the time  
"
Class {
	#name : #LapackLeastSquaresSolver,
	#superclass : #AbstractLinearRegressionSolver,
	#category : #'AI-Lapack-Experiment'
}

{ #category : #private }
LapackLeastSquaresSolver >> runRegressionSolverX: x y: y [

	^ AILinearRegressionLeastSquares new fitX: x y: y
]
