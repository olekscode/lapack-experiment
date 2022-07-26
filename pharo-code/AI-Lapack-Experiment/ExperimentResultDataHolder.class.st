"
I am a class that only store data to be transmitted
"
Class {
	#name : #ExperimentResultDataHolder,
	#superclass : #Object,
	#instVars : [
		'lapackLeastSquaresTime',
		'pharoGradientDescentTime'
	],
	#category : #'AI-Lapack-Experiment'
}

{ #category : #accessing }
ExperimentResultDataHolder >> lapackLeastSquaresTime [

	^ lapackLeastSquaresTime
]

{ #category : #accessing }
ExperimentResultDataHolder >> lapackLeastSquaresTime: anObject [

	lapackLeastSquaresTime := anObject
]

{ #category : #accessing }
ExperimentResultDataHolder >> pharoGradientDescentTime [

	^ pharoGradientDescentTime
]

{ #category : #accessing }
ExperimentResultDataHolder >> pharoGradientDescentTime: anObject [

	pharoGradientDescentTime := anObject
]

{ #category : #printing }
ExperimentResultDataHolder >> printOn: aStream [

	aStream
		<< 'Lapack time ' << lapackLeastSquaresTime asString << String cr
		<< 'Pharo Gradient Descent Time: ' << pharoGradientDescentTime asString
]
