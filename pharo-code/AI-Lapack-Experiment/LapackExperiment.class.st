"
I am the class that runs the experiment for all the linear regression solvers and store the benchmarkings into a csv file.
"
Class {
	#name : #LapackExperiment,
	#superclass : #Object,
	#category : #'AI-Lapack-Experiment'
}

{ #category : #constants }
LapackExperiment class >> bigDatasetFile [

	^ self dataDirectory / 'big_dataset.csv'
]

{ #category : #constants }
LapackExperiment class >> dataDirectory [

	^ LapackExperimentGitBridge root / 'data'
]

{ #category : #constants }
LapackExperiment class >> mediumDatasetFile [

	^ self dataDirectory / 'medium_dataset.csv'
]

{ #category : #constants }
LapackExperiment class >> smallDatasetFile [

	^ self dataDirectory / 'small_dataset.csv'
]

{ #category : #utilities }
LapackExperiment >> loadData: aFile [

	| data asNumbers x y |

	data := (NeoCSVReader on: aFile readStream) upToEnd.

	asNumbers := data collect: [ :each | 
		each collect: [ :other | other asNumber ] ].

	x := asNumbers collect: [ :each | each allButLast ].
	y := asNumbers collect: [ :each | each last ].

	^ Dictionary new
		at: 'x' put: x;
		at: 'y' put: y;
		yourself
]

{ #category : #actions }
LapackExperiment >> outputResultSmall: resultsOfSmallDataset medium: resultsOfMediumDataset big: resultsOfBigDataset [

	self printResultsSmall: resultsOfSmallDataset
		medium: resultsOfMediumDataset
		big: resultsOfBigDataset
]

{ #category : #actions }
LapackExperiment >> printResultsSmall: resultsOfSmallDataset medium: resultsOfMediumDataset big: resultsOfBigDataset [


	| text |
	text := String streamContents: [ :aStream |
		aStream 
			<< 'Small dataset' << String cr << resultsOfSmallDataset asString << String cr << String cr
			<< 'Medium dataset' << String cr << resultsOfMediumDataset asString << String cr << String cr
			<< 'Big dataset' << String cr << resultsOfBigDataset asString ].
	
	SpTextPresenter new
		text: text;
		open.
		
	^ text
]

{ #category : #actions }
LapackExperiment >> runBigExperimentLapack: lapackSolver pharoGradientDescent: gradientDescentSolver [

	| data lapackTime gradientDescentTime |
	
	data := self loadData: self class bigDatasetFile.

	lapackTime := lapackSolver runOnData: data.
	"Takes too long, described in the paper"
	gradientDescentTime := '-'.

	^ ExperimentResultDataHolder new
		lapackLeastSquaresTime: lapackTime;
		pharoGradientDescentTime: gradientDescentTime
]

{ #category : #api }
LapackExperiment >> runExperiment [

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
LapackExperiment >> runMediumExperimentLapack: lapackSolver pharoGradientDescent: gradientDescentSolver [

	| data lapackTime gradientDescentTime |
	
	data := self loadData: self class mediumDatasetFile.

	lapackTime := lapackSolver runOnData: data.
	gradientDescentTime := gradientDescentSolver runOnData: data.
	
	^ ExperimentResultDataHolder new
		lapackLeastSquaresTime: lapackTime;
		pharoGradientDescentTime: gradientDescentTime
]

{ #category : #actions }
LapackExperiment >> runSmallExperimentLapack: lapackSolver pharoGradientDescent: gradientDescentSolver [

	| data lapackTime gradientDescentTime |
	
	data := self loadData: self class smallDatasetFile.

	lapackTime := lapackSolver runOnData: data.
	gradientDescentTime := gradientDescentSolver runOnData: data.

	^ ExperimentResultDataHolder new
		lapackLeastSquaresTime: lapackTime;
		pharoGradientDescentTime: gradientDescentTime
]
