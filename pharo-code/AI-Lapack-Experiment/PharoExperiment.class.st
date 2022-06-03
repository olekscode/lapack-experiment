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

	^ './pharo-local/iceberg/YOUR-NAME/lapack-experiment/experiment-results/pharo-results.csv'
]

{ #category : #actions }
PharoExperiment >> runBigExperimentLapack: lapackSolver pharoGradientDescent: gradientDescentSolver [

	| lapackBigTime gradientDescentBigTime |

	lapackBigTime := lapackSolver runBigDatasetExperiment.
	Transcript crShow: 'Finished lapack big dataset in ', lapackBigTime asString ,' seconds'.
	gradientDescentBigTime := '-'.
	Transcript crShow: 'Finished gradient descent big dataset in ', gradientDescentBigTime asString ,' seconds'.
	
	^ ExperimentResultDataHolder new
		lapackLeastSquaresTime: lapackBigTime;
		pharoGradientDescentTime: gradientDescentBigTime
]

{ #category : #api }
PharoExperiment >> runExperiment [

	self runExperimentForAllSolvers
]

{ #category : #actions }
PharoExperiment >> runExperimentForAllSolvers [

	| resultsOfSmallDataset resultsOfMediumDataset resultsOfBigDataset lapackSolver gradientDescentSolver |
	lapackSolver := LapackLeastSquaresSolver new.
	gradientDescentSolver := PharoGradientDescentSolver new.

	"Small experiment"
	resultsOfSmallDataset := self runSmallExperimentLapack: lapackSolver pharoGradientDescent: gradientDescentSolver.

	"Medium experiment"
	resultsOfMediumDataset := self runMediumExperimentLapack: lapackSolver pharoGradientDescent: gradientDescentSolver.
	
	"Big experiment"
	resultsOfBigDataset := self runBigExperimentLapack: lapackSolver pharoGradientDescent: gradientDescentSolver.

	self writeCsvFileToDiskSmall: resultsOfSmallDataset
		medium: resultsOfMediumDataset
		big: resultsOfBigDataset
]

{ #category : #actions }
PharoExperiment >> runMediumExperimentLapack: lapackSolver pharoGradientDescent: gradientDescentSolver [

	| lapackMediumlTime gradientDescentMediumTime |

	lapackMediumlTime := lapackSolver runMediumDatasetExperiment.
	Transcript crShow: 'Finished lapack medium dataset in ', lapackMediumlTime asString ,' seconds'.
	gradientDescentMediumTime := gradientDescentSolver runMediumDatasetExperiment.
	Transcript crShow: 'Finished gradient descent medium dataset in ', gradientDescentMediumTime asString ,' seconds'.
	
	^ ExperimentResultDataHolder new
		lapackLeastSquaresTime: lapackMediumlTime;
		pharoGradientDescentTime: gradientDescentMediumTime
]

{ #category : #actions }
PharoExperiment >> runSmallExperimentLapack: lapackSolver pharoGradientDescent: gradientDescentSolver [

	| lapackSmallTime gradientDescentSmallTime |

	lapackSmallTime := lapackSolver runSmallDatasetExperiment.
	Transcript crShow: 'Finished lapack small dataset in ', lapackSmallTime asString ,' seconds'.
	gradientDescentSmallTime := gradientDescentSolver runSmallDatasetExperiment.
	Transcript crShow: 'Finished gradient descent small dataset in ', gradientDescentSmallTime asString ,' seconds'.

	^ ExperimentResultDataHolder new
		lapackLeastSquaresTime: lapackSmallTime;
		pharoGradientDescentTime: gradientDescentSmallTime
]

{ #category : #actions }
PharoExperiment >> writeCsvFileToDiskSmall: resultsOfSmallDataset medium: resultsOfMediumDataset big: resultsOfBigDataset [

	| file |
	file := self csvFile.
	file writeStreamDo: [ :stream | 
		stream
			<< 'Pharo + LAPACK Least Squares, ' << 'Pure Pharo Gradient Descent'
			<< String cr.
		resultsOfSmallDataset printOn: stream.
		stream << String cr.
		resultsOfMediumDataset printOn: stream.
		stream << String cr.
		resultsOfBigDataset printOn: stream ]
]
