Class {
	#name : #LapackExperimentGitBridge,
	#superclass : #GitBridge,
	#category : #'AI-Lapack-Experiment'
}

{ #category : #'class initialization' }
LapackExperimentGitBridge class >> initialize [

	SessionManager default registerSystemClassNamed: self name
]
