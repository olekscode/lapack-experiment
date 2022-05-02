Class {
	#name : #PharoGradientDescentExperiment,
	#superclass : #Object,
	#category : #'AI-Lapack-Experiment'
}

{ #category : #'memory space' }
PharoGradientDescentExperiment class >> garbageCollect [

	^ 10 timesRepeat: [ SmalltalkImage current garbageCollect ]
]

{ #category : #utilities }
PharoGradientDescentExperiment class >> loadData: fileName [

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

{ #category : #api }
PharoGradientDescentExperiment class >> runSmallDatasetExperiment [

	| data gradientDescentModel lapackTime x y |
	data := self loadData: 'feynman_I_10_7.csv'.
	x := data at: 'x'.
	y := data at: 'y'.

	gradientDescentModel := AILinearRegression new.

	self garbageCollect.
	
	lapackTime := [ gradientDescentModel fitX: x y: y ] timeToRun.

	^ lapackTime
]
