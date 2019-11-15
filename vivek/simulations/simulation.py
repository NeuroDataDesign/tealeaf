from itertools import repeat
from multiprocessing import Pool

import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def sample(n_features, n_targets, epsilon, n_samples=1000, random_state=715, p_informative=0.5):

    n_informative = round(n_features * p_informative)
    X, y, coef = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_targets=n_targets,
        random_state=random_state,
        coef=True,
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.15)

    # Contaminate a subset of the data
    n_contaminated = int(round(epsilon * n_samples))
    contaminated = np.random.choice(range(y_train.shape[1]), size=n_contaminated)
    y_train[contaminated] = np.square(y_train[contaminated])

    return X_train, X_test, y_train, y_test


def _train_forest(X, y, criterion):
    """
    Fit a RandomForestRegressor with default parameters and specific criterion.
    """

    regr = RandomForestRegressor(n_estimators=500, criterion=criterion)
    regr.fit(X, y)
    return regr


def _test_forest(X, y, regr):
    """
    Calculate the accuracy of the model on a heldout set.
    """

    y_pred = regr.predict(X)
    return norm(y_pred - y) / len(y)


def main(n_features, n_targets, epsilon, n_iter=25):

    print(epsilon)

    scores = []

    for _ in range(n_iter):

        # Sample training and testing data
        X_train, X_test, y_train, y_test = sample(n_features, n_targets, epsilon)

        # Train forests and score them
        criteria = ["mae", "mse", "friedman_mse"]
        score = []
        for criterion in criteria:
            regr = _train_forest(X_train, y_train, criterion)
            forest_score = _test_forest(X_test, y_test, regr)
            score.append(forest_score)

        scores.append(score)

    scores = np.array(scores)
    return scores.mean(axis=0)


if __name__ == "__main__":

    with Pool() as pool:

        scores = pool.starmap(
            main, zip(repeat(6), repeat(2), np.linspace(0, 0.5, num=50))
        )

        param_space = zip(repeat(6), repeat(2), np.linspace(0, 0.5, num=50))
        param_space = np.array(list(param_space))

        df = np.concatenate((param_space, scores), axis=1)
        columns = ["n_features", "n_targets", "epsilon", "mae", "mse", "friedman_mse"]
        df = pd.DataFrame(df, columns=columns)
        print(df.head())

        df.to_csv("sim_results.csv")
