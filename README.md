# lapack-experiment

## Installation

In order to install the experiment, perform the following code in a Playground:

```st
Metacello new
    repository: 'github://jordanmontt/lapack-experiment';
    baseline: 'AILapackExperiment';
    load
```

## Run the experiment

1. Clone this repository using the `Metacello` script.
2. Decompress the file `1191_BNG_pbc.tsv.gz` that is located in `'./pharo-local/iceberg/jordanmontt/lapack-experiment/experiment-files'` file location.
3. Run the python file `convert-1191_BNG_pbc to csv.py` to convert the file into an `csv`.

Then you are all set.

- `LapackLeastSquaresExperiment` solves the linear regression problem using the lapack least squares algorithm.
- `PharoLeastSquaresExperiment` solves the linear regression problem using the Pharo least squares algorithm.
- `PharoGradientDescentExperiment` solves the linear regression problem using the Pharo gradient descent algorithm.

All the classes have the same API. You can send this class side messages to any of the experiment classes. The methods will return the time that the algorithms takes to be solved.

- `runMediumDatasetExperiment`
- `runSmallDatasetExperiment`

For example:

```st
timeToRunLapackLeastSquaresMedium := LapackLeastSquaresExperiment runMediumDatasetExperiment.

timeToRunPharoGradientDescentSmall := PharoGradientDescentExperiment runSmallDatasetExperiment.

timeToRunPharoLeastSquaresSmall := PharoLeastSquaresExperiment runSmallDatasetExperiment.
```
