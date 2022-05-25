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
	file deleteIfAbsent: [  ].
	file ensureCreateFile.
	^ file
]

{ #category : #actions }
PharoExperiment >> fileName [

	^ './pharo-local/iceberg/jordanmontt/lapack-experiment/experiment-results/pharo-results-artificial.csv'
]

{ #category : #actions }
PharoExperiment >> runBigExperimentLapack: lapackSolver gradientD: gradientDescentSolver leastS: leastSquaresSolver [

	| lapackBigTime gradientDescentBigTime leastSquaresBigTime |

	lapackBigTime := lapackSolver runBigDatasetExperiment.
	Transcript crShow: 'Finished lapack big dataset in ', lapackBigTime asString ,' seconds'.
	gradientDescentBigTime := "gradientDescentSolver runBigDatasetExperiment" '-'.
	Transcript crShow: 'Finished gradient descent big dataset in ', gradientDescentBigTime asString ,' seconds'.
	leastSquaresBigTime := "leastSquaresSolver runBigDatasetExperiment" '-'.
	Transcript crShow: 'Finished least squares big dataset in ', leastSquaresBigTime asString ,' seconds'.

	^ ExperimentResultDataHolder new
		lapackLeastSquaresTime: lapackBigTime;
		pharoGradientDescentTime: gradientDescentBigTime;
		pharoLeastSquaresTime: leastSquaresBigTime
]

{ #category : #api }
PharoExperiment >> runExperiment [

	self runExperimentForAllSolvers
]

{ #category : #actions }
PharoExperiment >> runExperimentForAllSolvers [

	| resultsOfSmallDataset resultsOfMediumDataset resultsOfBigDataset lapackSolver gradientDescentSolver pharoLeastSquaresSolver |
	lapackSolver := LapackLeastSquaresSolver new.
	gradientDescentSolver := PharoGradientDescentSolver new.
	pharoLeastSquaresSolver := PharoLeastSquaresSolver new.

	"Small experiment"
	resultsOfSmallDataset := self runSmallExperimentLapack: lapackSolver gradientD: gradientDescentSolver leastS: pharoLeastSquaresSolver.

	"Medium experiment"
	resultsOfMediumDataset := self runMediumExperimentLapack: lapackSolver gradientD: gradientDescentSolver leastS: pharoLeastSquaresSolver.
	
	"Big experiment"
	resultsOfBigDataset := self runBigExperimentLapack: lapackSolver gradientD: gradientDescentSolver leastS: pharoLeastSquaresSolver.

	self writeCsvFileToDiskSmall: resultsOfSmallDataset
		medium: resultsOfMediumDataset
		big: resultsOfBigDataset
]

{ #category : #actions }
PharoExperiment >> runMediumExperimentLapack: lapackSolver gradientD: gradientDescentSolver leastS: leastSquaresSolver [

	| lapackMediumlTime gradientDescentMediumTime leastSquaresMediumTime |

	lapackMediumlTime := lapackSolver runMediumDatasetExperiment.
	Transcript crShow: 'Finished lapack medium dataset in ', lapackMediumlTime asString ,' seconds'.
	gradientDescentMediumTime := "gradientDescentSolver runMediumDatasetExperiment" '-'.
	Transcript crShow: 'Finished gradient descent medium dataset in ', gradientDescentMediumTime asString ,' seconds'.
	leastSquaresMediumTime := "leastSquaresSolver runMediumDatasetExperiment" '-'.
	Transcript crShow: 'Finished least squares medium dataset in ', leastSquaresMediumTime asString ,' seconds'.

	^ ExperimentResultDataHolder new
		lapackLeastSquaresTime: lapackMediumlTime;
		pharoGradientDescentTime: gradientDescentMediumTime;
		pharoLeastSquaresTime: leastSquaresMediumTime
]

{ #category : #actions }
PharoExperiment >> runSmallExperimentLapack: lapackSolver gradientD: gradientDescentSolver leastS: leastSquaresSolver [

	| lapackSmallTime gradientDescentSmallTime leastSquaresSmallTime |

	lapackSmallTime := lapackSolver runSmallDatasetExperiment.
	Transcript crShow: 'Finished lapack small dataset in ', lapackSmallTime asString ,' seconds'.
	gradientDescentSmallTime := gradientDescentSolver runSmallDatasetExperiment.
	Transcript crShow: 'Finished gradient descent small dataset in ', gradientDescentSmallTime asString ,' seconds'.
	leastSquaresSmallTime := leastSquaresSolver runSmallDatasetExperiment.
	Transcript crShow: 'Finished least squares small dataset in ', leastSquaresSmallTime asString ,' seconds'.

	^ ExperimentResultDataHolder new
		lapackLeastSquaresTime: lapackSmallTime;
		pharoGradientDescentTime: gradientDescentSmallTime;
		pharoLeastSquaresTime: leastSquaresSmallTime
]

{ #category : #actions }
PharoExperiment >> writeCsvFileToDiskSmall: resultsOfSmallDataset medium: resultsOfMediumDataset big: resultsOfBigDataset [

	| file |
	file := self csvFile.
	file writeStreamDo: [ :stream | 
		stream
			<< 'Pharo + LAPACK Least Squares, ' << 'Pure Pharo Least Squares, ' << 'Pure Pharo Gradient Descent'
			<< String cr.
		resultsOfSmallDataset printOn: stream.
		stream << String cr.
		resultsOfMediumDataset printOn: stream.
		stream << String cr.
		resultsOfBigDataset printOn: stream ]
]
