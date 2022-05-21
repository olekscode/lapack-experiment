"
I am a class that only store data to be transmitted
"
Class {
	#name : #ExperimentResultDataHolder,
	#superclass : #Object,
	#instVars : [
		'lapackLeastSquaresTime',
		'pharoGradientDescentTime',
		'pharoLeastSquaresTime'
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

{ #category : #accessing }
ExperimentResultDataHolder >> pharoLeastSquaresTime [

	^ pharoLeastSquaresTime
]

{ #category : #accessing }
ExperimentResultDataHolder >> pharoLeastSquaresTime: anObject [

	pharoLeastSquaresTime := anObject
]

{ #category : #printing }
ExperimentResultDataHolder >> printOn: aStream [

	aStream << lapackLeastSquaresTime asString << ','
	<< pharoLeastSquaresTime asString << ','
	<< pharoGradientDescentTime asString
]
