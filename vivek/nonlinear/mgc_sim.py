from itertools import product
from multiprocessing import Pool
import time

import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.ensemble import RandomForestRegressor

from mgcpy.benchmarks.simulations import (
    joint_sim,
    sin_sim,
    sin_sim,
    multi_noise_sim,
    step_sim,
    spiral_sim,
    circle_sim,
    circle_sim,
    square_sim,
    log_sim,
    quad_sim,
    w_sim,
    two_parab_sim,
    root_sim,
    multi_indep_sim,
    ubern_sim,
    square_sim,
    linear_sim,
    exp_sim,
    cub_sim,
)


def _train_forest(X, y, criterion):
    """
    Fit a RandomForestRegressor with default parameters and specific criterion.
    """

    if y.shape[1] == 1:
        y = np.ravel(y)

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


simulations = {
    "joint_normal": (joint_sim, 4),
    "sine_4pi": (sin_sim, 12),
    "sine_16pi": (sin_sim, 13),
    "multi_noise": (multi_noise_sim, 19),
    "step": (step_sim, 5),
    "spiral": (spiral_sim, 8),
    "circle": (circle_sim, 16),
    "ellipse": (circle_sim, 17),
    "diamond": (square_sim, 18),
    "log": (log_sim, 10),
    "quadratic": (quad_sim, 6),
    "w_shape": (w_sim, 7),
    "two_parabolas": (two_parab_sim, 15),
    "fourth_root": (root_sim, 11),
    "multi_indept": (multi_indep_sim, 20),
    "bernoulli": (ubern_sim, 9),
    "square": (square_sim, 14),
    "linear": (linear_sim, 1),
    "exponential": (exp_sim, 2),
    "cubic": (cub_sim, 3),
}


def find_dim(sim_name):
    if sim_name in ["joint_normal", "sine_4pi", "sine_16pi", "multi_noise"]:
        dim = 10
    elif sim_name in [
        "step",
        "spiral",
        "circle",
        "ellipse",
        "quadratic",
        "w_shape",
        "two_parabolas",
        "fourth_root",
    ]:
        dim = 20
    elif sim_name in ["multi_indept", "bernoulli", "log"]:
        dim = 100
    elif sim_name in ["linear", "exponential", "cubic"]:
        dim = 1000
    else:
        dim = 40
    return dim


def main(n_train, sim_name, criterion, n_iter=5):

    # Print the simulation name
    sim, _ = simulations[sim_name]
    # dim = find_dim(sim_name)
    dim = 10

    # Make a validation dataset
    X_test, y_test = sim(num_samp=1000, num_dim=dim)

    # Train forests and score them
    score = []
    for _ in range(n_iter):

        X_train, y_train = sim(num_samp=int(n_train), num_dim=dim)
        regr = _train_forest(X_train, y_train, criterion)
        forest_score = _test_forest(X_test, y_test, regr)
        score.append(forest_score)

    # Calculate average and standard deviation
    score = np.array(score)
    average = score.mean()
    error = score.std() / np.sqrt(n_iter)
    out = np.array([average, error])

    np.savetxt(f"results/sim_2/{sim_name}_{criterion}_{n_train}", out)
    print(sim_name)

    return out


if __name__ == "__main__":

    # Start running the simulations
    start_time = time.time()

    # Save parameter space as a numpy array
    params = product(
        range(5, 105, 5), simulations.keys(), ["mae", "mse", "friedman_mse"]
    )
    params = np.array(list(params))

    # Open multiprocessing
    with Pool() as pool:

        # Run the pools
        scores = pool.starmap(main, params)

        # Save data to array
        df = np.concatenate((params, scores), axis=1)
        columns = ["n_samples", "simulation", "criterion", "average", "error"]
        df = pd.DataFrame(df, columns=columns)
        print(df.head())

        df.to_csv("./results/sim_2/sim_2.csv")

    # Print runtime
    print("All finished!")
    print("Took {} minutes".format((time.time() - start_time) / 60))
