import numpy as np
import pandas as pd
from scipy.stats import ortho_group
from tqdm import tqdm

from rf import RandomForest


def generate_random_data(n_samples, n_dim, mean=None, cov=None):
    """
    Sample random input and output data for regression simulations.

    The input data, X, is sampled iid from a multivariate normal distribution.
    The output data, y, is produced by rotating the input data.
    The rotation matrix is randomly sampled from the O(N) Haar distribution.

    Parameters
    ==========
    n_samples : int
        Number of samples to generate
    n_dim : int
        Number of dimensions
    mean : array-like, shape=n_dim
        Mean of the multivariate normal distribution
    cov : array-like, shape=(n_dim, n_dim)
        Covariance matrix of the multivariate normal distribution

    Returns
    =======
    X : array-like, shape=(n_samples, n_dim)
    y : array-like, shape=(n_samples, n_dim)
    """

    # Handle erroneous arguments
    if mean is not None:
        assert n_dim == len(mean), "len(mean) must be equal to n_dim"
    if cov is not None:
        assert (
            n_dim == cov.shape[0] == cov.shape[1]
        ), "cov must be a square matrix with shape (n_dim, n_dim)"

    # Handle missing arguments
    if mean is None:
        mean = np.zeros(n_dim)
        mean[0] = 1
    if cov is None:
        cov = np.identity(n_dim)

    # Sample a random rotation matrix
    A = ortho_group.rvs(n_dim)

    # Generate the input and output data (X, y)
    X = np.random.multivariate_normal(mean, cov, size=n_samples)
    y = np.dot(A, X.T).T

    return X, y


def measure_mse(
    X, y, max_depth=10, n_features=1, min_leaf_size=5, n_trees=1000, n_bagging=10
):
    """
    For each split criteria, measure Mean Squared Error (MSE).

    Parameters
    ==========
    X : array_like
    y : array_like
    max_depth : int
    n_features : int
    min_leaf_size : int
    n_trees : int
    n_bagging : int

    Returns
    =======
    errors : list
        List of MSEs for different split criteria
    """

    # Make a dictionary of default parameters
    default = {
        "max_depth": max_depth,
        "n_features": n_features,
        "min_leaf_size": min_leaf_size,
        "n_trees": n_trees,
        "n_bagging": n_bagging,
    }

    # Iterate over different split criteria and calculate MSE
    n_samples = X.shape[0]
    errors = []
    for split in ["mae", "mse", "projection_axis", "projection_random"]:

        # Fit model
        rf = RandomForest(criteria=split, **default)
        rf.fit(X, y)

        # Make predictions and score
        yhat = rf.predict(X)
        mse = np.linalg.norm(y - yhat) / n_samples
        errors.append(mse)

    return errors


def run_simulation(simulation_params, rf_params={}):

    columns = ["n_samples", "n_dim", "mae", "mse", "axis", "oblique"]
    results = []

    for n_samples in tqdm(simulation_params["n_samples"], desc="Number of samples"):
        for n_dim in tqdm(simulation_params["n_dim"], desc="Number of dimensions"):
            for _ in tqdm(range(simulation_params["n_iter"]), desc="Iterations"):

                X, y = generate_random_data(n_samples, n_dim)
                mse = measure_mse(X, y, **rf_params)
                results.append([n_samples, n_dim] + mse)

    # Melt dataframe
    results = pd.DataFrame(results, columns=columns)
    results = pd.melt(
        results,
        id_vars=["n_samples", "n_dim"],
        value_vars=columns[2:],
        var_name="split",
        value_name="mse",
    )

    return results