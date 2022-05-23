"
I am the class that runs the experiment for all the linear regression solvers and store the benchmarkings into a csv file.
"
Class {
	#name : #PharoExperiment,
	#superclass : #Object,
	#category : #'AI-Lapack-Experiment'
}

{ #category : #actions }
PharoExperiment >> csvFile [

	| file |
	file := self fileName asFileReference.
	file ensureCreateFile.
	^ file
]

{ #category : #actions }
PharoExperiment >> fileName [

	^ './pharo-local/iceberg/jordanmontt/lapack-experiment/experiment-results/pharo-results.csv'
]

{ #category : #api }
PharoExperiment >> runExperiment [

	^ self runExperimentForAllSolvers
]

{ #category : #actions }
PharoExperiment >> runExperimentForAllSolvers [

	| resultsOfSmallDataset resultsOfMediumDataset resultsOfBigDataset lapackSolver gradientDescentSolver pharoLeastSquaresSolver |
	
	lapackSolver := LapackLeastSquaresSolver new.
	gradientDescentSolver := PharoGradientDescentSolver new.
	pharoLeastSquaresSolver := PharoLeastSquaresSolver new.
	
	resultsOfSmallDataset := ExperimentResultDataHolder new
		lapackLeastSquaresTime: lapackSolver runSmallDatasetExperiment;
		pharoGradientDescentTime:" gradientDescentSolver runSmallDatasetExperiment"'-';
		pharoLeastSquaresTime: "pharoLeastSquaresSolver runSmallDatasetExperiment"'-'.
	
	resultsOfMediumDataset := ExperimentResultDataHolder new
		lapackLeastSquaresTime: lapackSolver runMediumDatasetExperiment;
		pharoGradientDescentTime: "gradientDescentSolver runMediumDatasetExperiment"'-';
		pharoLeastSquaresTime: "pharoLeastSquaresSolver runMediumDatasetExperiment"'-'.
		
	"resultsOfBigDataset := ExperimentResultDataHolder new
		lapackLeastSquaresTime: lapackSolver runBigDatasetExperiment;
		pharoGradientDescentTime: gradientDescentSolver runBigDatasetExperiment;
		pharoLeastSquaresTime: pharoLeastSquaresSolver runBigDatasetExperiment."
	resultsOfBigDataset := ExperimentResultDataHolder new
		lapackLeastSquaresTime: '-';
		pharoGradientDescentTime: '-';
		pharoLeastSquaresTime: '-'.
	
	^ self writeCsvFileToDiskSmall: resultsOfSmallDataset medium: resultsOfMediumDataset big: resultsOfBigDataset
]

{ #category : #actions }
PharoExperiment >> writeCsvFileToDiskSmall: resultsOfSmallDataset medium: resultsOfMediumDataset big: resultsOfBigDataset [

	| file |
	file := self csvFile.
	^ file writeStreamDo: [ :stream | 
		stream
			<< 'Pharo + LAPACK Least Squares, ' << 'Pure Pharo Least Squares, ' << 'Pure Pharo Gradient Descent'
			<< String cr.
		resultsOfSmallDataset printOn: stream.
		stream << String cr.
		resultsOfMediumDataset printOn: stream.
		stream << String cr.
		resultsOfBigDataset printOn: stream ]
]
