Class {
	#name : #AbstractExperiment,
	#superclass : #Object,
	#category : #'AI-Lapack-Experiment'
}

{ #category : #utilities }
AbstractExperiment class >> garbageCollect [

	^ 10 timesRepeat: [ SmalltalkImage current garbageCollect ]
]

{ #category : #'experiment definition' }
AbstractExperiment class >> linearRegressionSolvingAlgorithmClass [

	^ self subclassResponsibility
]

{ #category : #utilities }
AbstractExperiment class >> loadData: fileName [

	| directory file data withoutHeaders x y |
	directory := './pharo-local/iceberg/jordanmontt/lapack-experiment/experiment-files'.
	file := (directory , '/' , fileName) asFileReference.

	data := (NeoCSVReader on: file readStream) upToEnd.

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
AbstractExperiment class >> mediumDatasetFileName [

	^ '1191_BNG_pbc.csv'
]

{ #category : #private }
AbstractExperiment class >> runExperiment: datasetFileName [

	| data model lapackTime x y |
	data := self loadData: datasetFileName.
	x := data at: 'x'.
	y := data at: 'y'.

	model := self linearRegressionSolvingAlgorithmClass new.

	self garbageCollect.
	
	lapackTime := [ model fitX: x y: y ] timeToRun.

	^ lapackTime
]

{ #category : #api }
AbstractExperiment class >> runMediumDatasetExperiment [

	^ self runExperiment: self mediumDatasetFileName
]

{ #category : #api }
AbstractExperiment class >> runSmallDatasetExperiment [

	^ self runExperiment: self smallDatasetFileName
]

{ #category : #'experiment definition' }
AbstractExperiment class >> smallDatasetFileName [

	^ 'feynman_I_10_7.csv'
]
