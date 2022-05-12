Class {
	#name : #ExperimentResult,
	#superclass : #Object,
	#instVars : [
		'lapackLeastSquaresTime',
		'pharoGradientDescentTime',
		'pharoLeastSquaresTime'
	],
	#category : #'AI-Lapack-Experiment'
}

{ #category : #accessing }
ExperimentResult >> lapackLeastSquaresTime [

	^ lapackLeastSquaresTime
]

{ #category : #accessing }
ExperimentResult >> lapackLeastSquaresTime: anObject [

	lapackLeastSquaresTime := anObject
]

{ #category : #accessing }
ExperimentResult >> pharoGradientDescentTime [

	^ pharoGradientDescentTime
]

{ #category : #accessing }
ExperimentResult >> pharoGradientDescentTime: anObject [

	pharoGradientDescentTime := anObject
]

{ #category : #accessing }
ExperimentResult >> pharoLeastSquaresTime [

	^ pharoLeastSquaresTime
]

{ #category : #accessing }
ExperimentResult >> pharoLeastSquaresTime: anObject [

	pharoLeastSquaresTime := anObject
]

{ #category : #printing }
ExperimentResult >> printOn: aStream [

	aStream << lapackLeastSquaresTime asString << ','
	<< pharoLeastSquaresTime asString << ','
	<< pharoGradientDescentTime asString
]
