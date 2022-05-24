from sklearn.datasets import make_regression
import pandas
import numpy


def generate_dataset(number_rows, number_columns):
    x, y = make_regression(n_samples=number_rows, n_features=number_columns, n_informative=number_columns,
                           random_state=0)
    return list([(numpy.append(ex, ey)) for ex, ey in zip(x, y)])


def write_dataset_to_disk(dataset, dataset_name):
    df = pandas.DataFrame(dataset)
    path = './data' + dataset_name
    df.to_csv(path_or_buf=path, index=False, sep=',', header=False)


def run_dataset_generation(n_rows, n_columns, file_name):
    print('Starting generating dataset (', n_rows, ' rows x ', n_columns, ' columns )')
    dataset = generate_dataset(n_rows, n_columns)
    write_dataset_to_disk(dataset, file_name)
    print('Finished dataset generation')


# Small dataset

run_dataset_generation(100000, 5, 'small_dataset.csv')

# Medium dataset

run_dataset_generation(1000000, 20, 'medium_dataset.csv')

# Big dataset

run_dataset_generation(1000000, 300, 'big_dataset.csv')
