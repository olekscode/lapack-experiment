Class {
	#name : #LapackLeastSquaresExperiment,
	#superclass : #AbstractExperiment,
	#category : #'AI-Lapack-Experiment'
}

{ #category : #'experiment definition' }
LapackLeastSquaresExperiment class >> linearRegressionSolvingAlgorithmClass [

	^ AILinearRegressionLeastSquares
]
