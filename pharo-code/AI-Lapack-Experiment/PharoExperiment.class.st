"
I am the class that runs the experiment for all the linear regression solvers and store the benchmarkings into a csv file.
"
Class {
	#name : #PharoExperiment,
	#superclass : #Object,
	#category : #'AI-Lapack-Experiment'
}

{ #category : #actions }
PharoExperiment >> outputResultSmall: resultsOfSmallDataset medium: resultsOfMediumDataset big: resultsOfBigDataset [

	"self writeCsvFileToDiskSmall: resultsOfSmallDataset
		medium: resultsOfMediumDataset
		big: resultsOfBigDataset."
	
	self printResultsSmall: resultsOfSmallDataset medium: resultsOfMediumDataset big: resultsOfBigDataset
]

{ #category : #actions }
PharoExperiment >> printResultsSmall: resultsOfSmallDataset medium: resultsOfMediumDataset big: resultsOfBigDataset [


	| text |
	text := String streamContents: [ :aStream |
		aStream 
			<< 'Small dataset' << String cr << resultsOfSmallDataset asString << String cr << String cr
			<< 'Medium dataset' << String cr << resultsOfMediumDataset asString << String cr << String cr
			<< 'Big dataset' << String cr << resultsOfBigDataset asString  ].
	
	SpTextPresenter new
		text: text;
		open.
		
	^ text
]

{ #category : #actions }
PharoExperiment >> runBigExperimentLapack: lapackSolver pharoGradientDescent: gradientDescentSolver [

	| lapackBigTime gradientDescentBigTime |

	"lapackBigTime := lapackSolver runBigDatasetExperiment."
	lapackBigTime := '-'.
	"Takes too long, described in the paper"
	gradientDescentBigTime := '-'.

	^ ExperimentResultDataHolder new
		lapackLeastSquaresTime: lapackBigTime;
		pharoGradientDescentTime: gradientDescentBigTime
]

{ #category : #api }
PharoExperiment >> runExperiment [

	| resultsOfSmallDataset resultsOfMediumDataset resultsOfBigDataset lapackSolver gradientDescentSolver |
	lapackSolver := LapackLeastSquaresSolver new.
	gradientDescentSolver := PharoGradientDescentSolver new.

	"Small experiment"
	resultsOfSmallDataset := self runSmallExperimentLapack: lapackSolver pharoGradientDescent: gradientDescentSolver.

	"Medium experiment"
	resultsOfMediumDataset := self runMediumExperimentLapack: lapackSolver pharoGradientDescent: gradientDescentSolver.
	
	"Big experiment"
	resultsOfBigDataset := self runBigExperimentLapack: lapackSolver pharoGradientDescent: gradientDescentSolver.

	self outputResultSmall: resultsOfSmallDataset medium: resultsOfMediumDataset big: resultsOfBigDataset
]

{ #category : #actions }
PharoExperiment >> runMediumExperimentLapack: lapackSolver pharoGradientDescent: gradientDescentSolver [

	| lapackMediumlTime gradientDescentMediumTime |

	lapackMediumlTime := lapackSolver runMediumDatasetExperiment.
	"gradientDescentMediumTime := gradientDescentSolver runMediumDatasetExperiment."
	gradientDescentMediumTime := '-'.
	
	^ ExperimentResultDataHolder new
		lapackLeastSquaresTime: lapackMediumlTime;
		pharoGradientDescentTime: gradientDescentMediumTime
]

{ #category : #actions }
PharoExperiment >> runSmallExperimentLapack: lapackSolver pharoGradientDescent: gradientDescentSolver [

	| lapackSmallTime gradientDescentSmallTime |

	lapackSmallTime := lapackSolver runSmallDatasetExperiment.
	"gradientDescentSmallTime := gradientDescentSolver runSmallDatasetExperiment."
	gradientDescentSmallTime := '-'.

	^ ExperimentResultDataHolder new
		lapackLeastSquaresTime: lapackSmallTime;
		pharoGradientDescentTime: gradientDescentSmallTime
]
