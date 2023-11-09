"
I am the class that runs the experiment for all the linear regression solvers and store the benchmarkings into a csv file.
"
Class {
	#name : #LapackExperiment,
	#superclass : #Object,
	#instVars : [
		'results'
	],
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

{ #category : #initialization }
LapackExperiment >> initialize [

	super initialize.
	results := Dictionary new.
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
LapackExperiment >> runBigExperimentLapack: aLapackSolver pharoGradientDescent: aGradientDescentSolver [

	^ self
		runExperimentOnData: (self loadData: self class bigDatasetFile)
		lapackSolver: aLapackSolver
		gradientDescentSolver: aGradientDescentSolver
]

{ #category : #api }
LapackExperiment >> runExperiment [

	| lapackSolver gradientDescentSolver |
	
	lapackSolver := LapackLeastSquaresSolver new.
	gradientDescentSolver := PharoGradientDescentSolver new.

	"Small experiment"
	Transcript show: 'Running small experiment'; cr.
	results at: 'small' put: (self
		runSmallExperimentLapack: lapackSolver
		pharoGradientDescent: gradientDescentSolver).

	"Medium experiment"
	Transcript show: 'Running medium experiment'; cr.
	results at: 'medium' put: (self
		runMediumExperimentLapack: lapackSolver
		pharoGradientDescent: gradientDescentSolver).
	
	"Big experiment"
	"Transcript show: 'Running big experiment'; cr.
	results at: 'big' put: (self
		runBigExperimentLapack: lapackSolver
		pharoGradientDescent: gradientDescentSolver)."
]

{ #category : #actions }
LapackExperiment >> runExperimentOnData: aDataset lapackSolver: aLapackSolver gradientDescentSolver: aGradientDescentSolver [

	| lapackTime gradientDescentTime |

	lapackTime := aLapackSolver runOnData: aDataset.
	gradientDescentTime := aGradientDescentSolver runOnData: aDataset.

	^ ExperimentResultDataHolder new
		lapackLeastSquaresTime: lapackTime;
		pharoGradientDescentTime: gradientDescentTime
]

{ #category : #actions }
LapackExperiment >> runMediumExperimentLapack: aLapackSolver pharoGradientDescent: aGradientDescentSolver [
	
	^ self
		runExperimentOnData: (self loadData: self class mediumDatasetFile)
		lapackSolver: aLapackSolver
		gradientDescentSolver: aGradientDescentSolver
]

{ #category : #actions }
LapackExperiment >> runSmallExperimentLapack: aLapackSolver pharoGradientDescent: aGradientDescentSolver [

	^ self
		runExperimentOnData: (self loadData: self class smallDatasetFile)
		lapackSolver: aLapackSolver
		gradientDescentSolver: aGradientDescentSolver
]
