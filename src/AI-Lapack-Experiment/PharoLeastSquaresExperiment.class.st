Class {
	#name : #PharoLeastSquaresExperiment,
	#superclass : #AbstractExperiment,
	#category : #'AI-Lapack-Experiment'
}

{ #category : #'experiment definition' }
PharoLeastSquaresExperiment class >> linearRegressionSolvingAlgorithmClass [

	^ AILinearRegressionLeastSquaresVanilla
]
