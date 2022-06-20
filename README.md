# pharo-ai Lapack experiment

## Requirements

- Have installed `python3`, `scikit-learn`, `pandas` and `Pharo11` on your computer.
- Have around 7GB of free memory space in the hard drive for the datasets.

## Generating the datasets

To generate the datasets, you need to run the file `DatasetGenerator.py` using `python3`.

```
python3 DatasetGenerator.py
```

This step can take until 15 minutes because the datasets are big.

The script will generate the three datasets (small, medium and large) into the `data` directory.
Note that the datasets are generated using a random seed. So you always have exactly the same datasets.

## Running the Python benchmarks

To run the Python benchmarks, you need to execute the file `PythonExperiment.py` using `python3`.

The script will run the benchmarks and create a csv file with the results in the directory `./experiment-results/python-results.csv`

## Running the Pharo benchmarks

- First, you need to download the Pharo launcher from the Pharo website https://pharo.org/download
- Create a Pharo 11 image.
- Import the repository into the image using iceberg.
Then, override the class side method `PharoExperiment >> fileName` to return an string that will contain where to store the result csv file.

Finally, you need execute in a Playground

```st
PharoExperiment new runExperiment
```

That run the benchmarks and create a csv file with the results in the specified directory
