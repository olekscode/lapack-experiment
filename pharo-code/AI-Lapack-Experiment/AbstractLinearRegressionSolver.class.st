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

{ #category : #'experiment definition' }
AbstractLinearRegressionSolver >> bigDatasetFileName [

	^ 'big_dataset.csv'
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

	| directory file data asNumbers x y |
	directory := './pharo-local/iceberg/jordanmontt/lapack-experiment/data/'.
	file := (directory , '/' , fileName) asFileReference.

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

{ #category : #'experiment definition' }
AbstractLinearRegressionSolver >> mediumDatasetFileName [

	^ 'medium_dataset.csv'
]

{ #category : #api }
AbstractLinearRegressionSolver >> runBigDatasetExperiment [

	^ self runExperimentNTimes: (self loadData: self bigDatasetFileName)
]

{ #category : #private }
AbstractLinearRegressionSolver >> runExperimentNTimes: data [

	^ [ self runSolver: data ] on: Exception
		do: [ :e | e asString ]
]

{ #category : #api }
AbstractLinearRegressionSolver >> runMediumDatasetExperiment [

	^ self runExperimentNTimes: (self loadData: self mediumDatasetFileName)
]

{ #category : #private }
AbstractLinearRegressionSolver >> runRegressionSolverX: x y: y [

	^ self subclassResponsibility
]

{ #category : #api }
AbstractLinearRegressionSolver >> runSmallDatasetExperiment [

	^ self runExperimentNTimes: (self loadData: self smallDatasetFileName)
]

{ #category : #private }
AbstractLinearRegressionSolver >> runSolver: data [

	| timeAccumulator x y |
	x := data at: 'x'.
	y := data at: 'y'.

	self garbageCollect.

	timeAccumulator := [ 1 to: n do: [ :i | self runRegressionSolverX: x y: y ] ] timeToRun.

	^ (timeAccumulator / n) totalSeconds asFloat
]

{ #category : #'experiment definition' }
AbstractLinearRegressionSolver >> smallDatasetFileName [

	^ 'small_dataset.csv'
]
