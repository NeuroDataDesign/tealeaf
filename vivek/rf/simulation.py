import numpy as np
import pandas as pd
from mgcpy.benchmarks import simulations as sims
from scipy.stats import ortho_group
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from rf import RandomForest


def generate_linear_data(n_samples, n_dim, mean=None, cov=None, loc=0.0, scale=1.0):
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
    loc : float, default=0.0
        Mean of normally distributed noise
    scale : float, default=1.0
        Standard deviation of the normally distributed noise

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

    # Add Gaussian noise to the output data
    y += np.random.normal(loc=loc, scale=scale, size=y.shape)

    return X, y


def generate_nonlinear_data(sim, n_samples, n_dim, **params):

    sim = getattr(sims, sim)
    return sim(num_samp=n_samples, num_dim=n_dim, **params)


def measure_mse(
    X,
    y,
    max_depth=10,
    n_features=1,
    min_leaf_size=5,
    n_trees=1000,
    n_bagging=10,
    test_size=0.25,
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
    test_size : float, default=0.25
        Fraction of data to hold out for MSE evaluation

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

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Iterate over different split criteria and calculate MSE
    n_samples = X.shape[0]
    errors = []
    for split in ["mae", "mse", "projection_axis", "projection_random"]:

        # Fit model
        rf = RandomForest(criteria=split, **default)
        rf.fit(X_train, y_train)

        # Make predictions and score
        y_pred = rf.predict(X_test)
        mse = np.linalg.norm(y_test - y_pred) ** 2 / n_samples
        errors.append(mse)

    return errors


def to_csv(results, savename="simulation"):

    columns = ["n_samples", "n_dim", "mae", "mse", "axis", "oblique"]

    # Melt dataframe
    results = pd.DataFrame(results, columns=columns)
    results = pd.melt(
        results,
        id_vars=["n_samples", "n_dim"],
        value_vars=columns[2:],
        var_name="split",
        value_name="mse",
    )

    results.to_csv(f"results/{savename}.csv")


if __name__ == "__main__":

    # Simulation 1
    # Fix n_samples=100, increase n_dim.
    # -------------------------------------------------------------------------
    print("Simulation 1: Increasing dimensionality")
    n_samples = 75
    n_iter = 10

    results = []

    for n_dim in tqdm(
        np.arange(start=2, stop=40, step=1, dtype=np.int), desc="Number of dimensions"
    ):
        for _ in tqdm(range(n_iter), desc="Number of iterations"):
            X, y = generate_linear_data(n_samples, n_dim)
            mse = measure_mse(X, y)
            results.append([n_samples, n_dim] + mse)

    to_csv(results, savename="simulation_1")
    del results

    # Simulation 2
    # Fix n_samples=30, n_dim \in {3, 30}, increase noise.
    # -------------------------------------------------------------------------
    print("\nSimulation 2: Increase noise")
    n_samples = 30

    results = []

    for n_dim in tqdm([3, 30], desc="Number of dimensions"):
        for scale in tqdm(
            np.linspace(start=0, stop=10, num=50), desc="Number of noise steps"
        ):
            for _ in tqdm(range(n_iter), desc="Number of iterations"):
                X, y = generate_linear_data(n_samples, n_dim, scale=scale)
                mse = measure_mse(X, y)
                results.append([n_samples, n_dim] + mse)

    to_csv(results, savename="simulation_2")
    del results

    # Simulation 3
    # Try nonlinear simulations?
    # -------------------------------------------------------------------------
    print("\nSimulation 3: Try nonlinear simulations")
