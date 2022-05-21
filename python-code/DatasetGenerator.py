from sklearn.datasets import make_regression
import pandas

X_test, y_test = make_regression(n_samples=1000000, n_features=300, n_targets=0)

df = pandas.DataFrame(list(X_test))
df.to_csv(path_or_buf='../data/huge_dataset.csv', index=False, sep=',', header=False)
