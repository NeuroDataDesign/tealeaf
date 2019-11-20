from itertools import repeat
from multiprocessing import Pool

import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.ensemble import RandomForestRegressor

from gross import GrossErrorModel


def _train_forest(X, y, criterion):
    """
    Fit a RandomForestRegressor with default parameters and specific criterion.
    """

    regr = RandomForestRegressor(
        n_estimators=500, criterion=criterion, max_features="sqrt"
    )
    regr.fit(X, y)
    return regr


def _test_forest(X, y, regr):
    """
    Calculate the accuracy of the model on a heldout set.
    """

    y_pred = regr.predict(X)
    return norm(y_pred - y) / len(y)


def main(n_features, n_targets, epsilon, n_iter=50):

    print(epsilon)

    scores = []

    # Sample training and testing data
    gem = GrossErrorModel(
        n_features=n_features, n_targets=n_targets, n_informative=n_features
    )
    train_gen, X_test, y_test = gem.sample(
        epsilon, n_iter=n_iter, n_train=30, n_test=1000
    )

    criteria = ["mae", "mse", "friedman_mse"]

    # Train forests and score them
    for X_train, y_train in train_gen:

        score = []

        for criterion in criteria:
            regr = _train_forest(X_train, y_train, criterion)
            forest_score = _test_forest(X_test, y_test, regr)
            score.append(forest_score)

        scores.append(score)

    # Calculate average and standard deviation
    scores = np.array(scores)
    score = scores.mean(axis=0)
    error = scores.std(axis=0) / np.sqrt(n_iter)
    return np.concatenate((score, error))


if __name__ == "__main__":

    with Pool() as pool:

        scores = pool.starmap(
            main, zip(repeat(10), repeat(2), np.linspace(0, 0.5, num=10))
        )

        param_space = zip(repeat(10), repeat(2), np.linspace(0, 0.5, num=10))
        param_space = np.array(list(param_space))

        df = np.concatenate((param_space, scores), axis=1)
        columns = [
            "n_features",
            "n_targets",
            "epsilon",
            "mae_score",
            "mse_score",
            "friedman_mse_score",
            "mae_std",
            "mse_std",
            "friedman_mse_std",
        ]
        df = pd.DataFrame(df, columns=columns)
        print(df.head())

        df.to_csv("sim3_results.csv")
