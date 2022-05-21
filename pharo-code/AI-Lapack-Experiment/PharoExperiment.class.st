Class {
	#name : #PharoExperiment,
	#superclass : #Object,
	#category : #'AI-Lapack-Experiment'
}

{ #category : #actions }
PharoExperiment class >> csvFile [

	| file |
	file := self fileName asFileReference.
	file ensureCreateFile.
	^ file
]

{ #category : #actions }
PharoExperiment class >> fileName [

	^ './pharo-local/iceberg/jordanmontt/lapack-experiment/experiment-results/pharo-results.csv'
]

{ #category : #api }
PharoExperiment class >> runExperiment [

	^ self runExperimentForAllSolvers
]

{ #category : #actions }
PharoExperiment class >> runExperimentForAllSolvers [

	| resultsOfSmallDataset resultsOfMediumDataset |
	
	resultsOfSmallDataset := ExperimentResult new
		lapackLeastSquaresTime: LapackLeastSquaresLinearRegressionSolver runSmallDatasetExperiment;
		pharoGradientDescentTime: PharoGradientDescentLinearRegressionSolver runSmallDatasetExperiment;
		pharoLeastSquaresTime: "PharoLeastSquaresLinearRegressionSolver runSmallDatasetExperiment"'-'.
		
	resultsOfMediumDataset := ExperimentResult new
		lapackLeastSquaresTime: LapackLeastSquaresLinearRegressionSolver runMediumDatasetExperiment;
		pharoGradientDescentTime: PharoGradientDescentLinearRegressionSolver runMediumDatasetExperiment;
		pharoLeastSquaresTime: PharoLeastSquaresLinearRegressionSolver runMediumDatasetExperiment.
		
	^ self writeCsvFileToDiskSmall: resultsOfSmallDataset medium: resultsOfMediumDataset
]

{ #category : #actions }
PharoExperiment class >> writeCsvFileToDiskSmall: resultsOfSmallDataset medium: resultsOfMediumDataset [

	| file |
	file := self csvFile.
	^ file writeStreamDo: [ :stream | 
		stream
			<< 'Pharo + LAPACK Least Squares, ' << 'Pure Pharo Least Squares, ' << 'Pure Pharo Gradient Descent'
			<< String cr.
		resultsOfSmallDataset printOn: stream.
		stream << String cr.
		resultsOfMediumDataset printOn: stream ]
]
