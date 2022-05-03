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
			with: [ spec repository: 'github://pharo-ai/linear-models:lapack/src' ].
		spec
			baseline: 'NeoCSV'
			with: [ spec repository: 'github://svenvc/NeoCSV/repository' ].

		"Packages"
		spec
			package: 'AI-Lapack-Experiment'
			with: [ spec requires: #( 'AILinearModels' ) ] ]
]
