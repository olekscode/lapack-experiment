"
I am an abstract class for solving the linear regression using different algorithms
"
Class {
	#name : #AbstractLinearRegressionSolver,
	#superclass : #Object,
	#instVars : [
		'n'
	],
	#category : #'AI-Lapack-Experiment'
}

{ #category : #constants }
AbstractLinearRegressionSolver class >> bigDatasetFileName [

	^ 'big_dataset.csv'
]

{ #category : #constants }
AbstractLinearRegressionSolver class >> dataDirectory [

	^ './pharo-local/iceberg/jordanmontt/lapack-experiment/data/' asFileReference.
]

{ #category : #constants }
AbstractLinearRegressionSolver class >> mediumDatasetFileName [

	^ 'medium_dataset.csv'
]

{ #category : #constants }
AbstractLinearRegressionSolver class >> smallDatasetFileName [

	^ 'small_dataset.csv'
]

{ #category : #utilities }
AbstractLinearRegressionSolver >> garbageCollect [

	^ 10 timesRepeat: [ SmalltalkImage current garbageCollect ]
]

{ #category : #initialization }
AbstractLinearRegressionSolver >> initialize [

	super initialize.
	n := 1
]

{ #category : #utilities }
AbstractLinearRegressionSolver >> loadData: fileName [

	| file data asNumbers x y |

	file := self class dataDirectory / fileName.

	data := (NeoCSVReader on: file readStream) upToEnd.

	asNumbers := data collect: [ :each | 
		each collect: [ :other | other asNumber ] ].

	x := asNumbers collect: [ :each | each allButLast ].
	y := asNumbers collect: [ :each | each last ].

	^ Dictionary new
		at: 'x' put: x;
		at: 'y' put: y;
		yourself
]

{ #category : #api }
AbstractLinearRegressionSolver >> runBigDatasetExperiment [

	^ self runSolver: (self loadData: self class bigDatasetFileName)
]

{ #category : #api }
AbstractLinearRegressionSolver >> runMediumDatasetExperiment [

	^ self runSolver: (self loadData: self class mediumDatasetFileName)
]

{ #category : #private }
AbstractLinearRegressionSolver >> runRegressionSolverX: x y: y [

	^ self subclassResponsibility
]

{ #category : #api }
AbstractLinearRegressionSolver >> runSmallDatasetExperiment [

	^ self runSolver: (self loadData: self class smallDatasetFileName)
]

{ #category : #private }
AbstractLinearRegressionSolver >> runSolver: data [

	^ [ self runSolverNTimes: data ] on: Exception
		do: [ :e | e asString ]
]

{ #category : #private }
AbstractLinearRegressionSolver >> runSolverNTimes: data [

	| timeAccumulator x y |
	x := data at: 'x'.
	y := data at: 'y'.

	self garbageCollect.

	timeAccumulator := [ 1 to: n do: [ :i | self runRegressionSolverX: x y: y ] ] timeToRun.

	^ (timeAccumulator / n) totalSeconds asFloat
]
