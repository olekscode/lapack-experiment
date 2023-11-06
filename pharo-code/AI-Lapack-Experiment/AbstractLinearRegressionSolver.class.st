"
I am an abstract class for solving the linear regression using different algorithms
"
Class {
	#name : #AbstractLinearRegressionSolver,
	#superclass : #Object,
	#category : #'AI-Lapack-Experiment'
}

{ #category : #'as yet unclassified' }
AbstractLinearRegressionSolver class >> numberOfReproductions [
	"Number of times each experiment is going to be run.
	The results of all reproductions should be averaged"

	^ 1
]

{ #category : #utilities }
AbstractLinearRegressionSolver >> garbageCollect [

	^ 10 timesRepeat: [ SmalltalkImage current garbageCollect ]
]

{ #category : #private }
AbstractLinearRegressionSolver >> runOnData: data [

	^ [ self runOnData: data times: self class numberOfReproductions ]
		on: Exception
		do: [ :e | e asString ]
]

{ #category : #private }
AbstractLinearRegressionSolver >> runOnData: data times: aNumber [

	| timeAccumulator x y |
	x := data at: 'x'.
	y := data at: 'y'.

	self garbageCollect.

	timeAccumulator := [ 1 to: aNumber do: [ :i | self runRegressionSolverX: x y: y ] ] timeToRun.

	^ (timeAccumulator / aNumber) totalSeconds asFloat
]

{ #category : #private }
AbstractLinearRegressionSolver >> runRegressionSolverX: x y: y [

	^ self subclassResponsibility
]
