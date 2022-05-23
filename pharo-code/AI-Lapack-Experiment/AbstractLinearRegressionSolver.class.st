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

	^ 'huge_dataset.csv'
]

{ #category : #private }
AbstractLinearRegressionSolver >> experiment: data [

	| timeAccumulator x y |
	x := data at: 'x'.
	y := data at: 'y'.

	self garbageCollect.

	timeAccumulator := [ 1 to: n do: [ self runRegressionSolverX: x y: y ] ] timeToRun.

	^ timeAccumulator / n
]

{ #category : #utilities }
AbstractLinearRegressionSolver >> garbageCollect [

	^ 10 timesRepeat: [ SmalltalkImage current garbageCollect ]
]

{ #category : #initialization }
AbstractLinearRegressionSolver >> initialize [

	super initialize.
	n := 5
]

{ #category : #utilities }
AbstractLinearRegressionSolver >> loadData: fileName separator: aCharacter [

	| directory file data withoutHeaders x y |
	directory := './pharo-local/iceberg/jordanmontt/lapack-experiment/data/'.
	file := (directory , '/' , fileName) asFileReference.

	data := (NeoCSVReader on: file readStream)
		separator: aCharacter;
		upToEnd.

	withoutHeaders := data allButFirst collect: [ :each | 
		each collect: [ :other | other asNumber ] ].

	x := withoutHeaders collect: [ :each | each allButLast ].
	y := withoutHeaders collect: [ :each | each last ].

	^ Dictionary new
		at: 'x' put: x;
		at: 'y' put: y;
		yourself
]

{ #category : #'experiment definition' }
AbstractLinearRegressionSolver >> mediumDatasetFileName [

	^ '1191_BNG_pbc.tsv'
]

{ #category : #api }
AbstractLinearRegressionSolver >> runBigDatasetExperiment [

	^ self runExperimentNTimes: (self loadData: self bigDatasetFileName separator: $,)
]

{ #category : #private }
AbstractLinearRegressionSolver >> runExperimentNTimes: data [

	^ [ self experiment: data ] on: Exception
		  do: [ 'error' ]
]

{ #category : #api }
AbstractLinearRegressionSolver >> runMediumDatasetExperiment [

	^ self runExperimentNTimes: (self loadData: self mediumDatasetFileName separator: Character tab)
]

{ #category : #private }
AbstractLinearRegressionSolver >> runRegressionSolverX: x y: y [

	^ self subclassResponsibility
]

{ #category : #api }
AbstractLinearRegressionSolver >> runSmallDatasetExperiment [

	^ self runExperimentNTimes: (self loadData: self smallDatasetFileName separator: Character tab)
]

{ #category : #'experiment definition' }
AbstractLinearRegressionSolver >> smallDatasetFileName [

	^ 'feynman_I_10_7.tsv'
]
