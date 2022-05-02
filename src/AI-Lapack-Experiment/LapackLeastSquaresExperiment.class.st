Class {
	#name : #LapackLeastSquaresExperiment,
	#superclass : #Object,
	#category : #'AI-Lapack-Experiment'
}

{ #category : #utilities }
LapackLeastSquaresExperiment class >> garbageCollect [

	^ 10 timesRepeat: [ SmalltalkImage current garbageCollect ]
]

{ #category : #utilities }
LapackLeastSquaresExperiment class >> loadData: fileName [

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
LapackLeastSquaresExperiment class >> runMediumDatasetExperiment [

	| data lapackModel lapackTime x y |
	data := self loadData: '1191_BNG_pbc.csv'.
	x := data at: 'x'.
	y := data at: 'y'.

	lapackModel := AILinearRegressionLeastSquares new.

	self garbageCollect.
	
	lapackTime := [ 5 timesRepeat: [ lapackModel fitX: x y: y ] ] timeToRun.

	^ lapackTime / 5
]

{ #category : #api }
LapackLeastSquaresExperiment class >> runSmallDatasetExperiment [

	| data lapackModel lapackTime x y |
	data := self loadData: 'feynman_I_10_7.csv'.
	x := data at: 'x'.
	y := data at: 'y'.

	lapackModel := AILinearRegressionLeastSquares new.

	self garbageCollect.
	
	lapackTime := [ 10 timesRepeat: [ lapackModel fitX: x y: y ] ] timeToRun.

	^ lapackTime / 10
]
