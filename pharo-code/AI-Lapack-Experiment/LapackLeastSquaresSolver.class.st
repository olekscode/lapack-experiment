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

{ #category : #private }
LapackLeastSquaresSolver >> runSolverNTimes: data [

	| timeAccumulator x y xMatrix |
	x := data at: 'x'.
	y := data at: 'y'.
	
	xMatrix := AIColumnMajorMatrix rows: x.
	
	self garbageCollect.

	timeAccumulator := [ 1 to: n do: [ :i | self runRegressionSolverX: xMatrix y: y ] ] timeToRun.

	^ (timeAccumulator / n) totalSeconds asFloat
]
