"
I am a beseline
"
Class {
	#name : #BaselineOfAILapackExperiment,
	#superclass : #BaselineOf,
	#category : #BaselineOfAILapackExperiment
}

{ #category : #baselines }
BaselineOfAILapackExperiment >> baseline: spec [

	<baseline>
	spec for: #common do: [
		"External dependencies"
		spec
			baseline: 'AILinearModels'
			with: [ spec repository: 'github://pharo-ai/linear-models/src' ].
		spec
			baseline: 'NeoCSV'
			with: [ spec repository: 'github://svenvc/NeoCSV/repository' ].
		spec
			baseline: 'GitBridge'
			with: [ spec repository: 'github://jecisc/GitBridge:v1.x.x/src' ].

		"Packages"
		spec
			package: 'AI-Lapack-Experiment'
			with: [ spec requires: #(AILinearModels GitBridge) ] ]
]
