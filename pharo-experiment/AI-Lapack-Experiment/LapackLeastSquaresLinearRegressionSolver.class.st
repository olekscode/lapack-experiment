Class {
	#name : #LapackLeastSquaresLinearRegressionSolver,
	#superclass : #AbstractLinearRegressionSolver,
	#category : #'AI-Lapack-Experiment'
}

{ #category : #'experiment definition' }
LapackLeastSquaresLinearRegressionSolver class >> linearRegressionSolvingAlgorithmClass [

	^ AILinearRegressionLeastSquares
]
