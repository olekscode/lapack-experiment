Class {
	#name : #PharoLeastSquaresLinearRegressionSolver,
	#superclass : #AbstractLinearRegressionSolver,
	#category : #'AI-Lapack-Experiment'
}

{ #category : #'experiment definition' }
PharoLeastSquaresLinearRegressionSolver class >> linearRegressionSolvingAlgorithmClass [

	^ AILinearRegressionLeastSquaresVanilla
]
