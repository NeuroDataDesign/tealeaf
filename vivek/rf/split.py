import numpy as np


def mae(y):
    """
    Calcualte the mean absolute error
    """
    y_bar = np.mean(y, axis=0).reshape(-1, 1)
    return np.sum([np.abs(row - y_bar) for row in y]) / len(y)


def mse(y):
    """
    Calculate the mean squared error
    """
    y_bar = np.mean(y, axis=0).reshape(-1, 1)
    return np.sum([np.linalg.norm(row - y_bar) ** 2 for row in y]) / len(y)


def projection_axis(y):
    """
    Project high dimensional y onto an axis and split.
    """

    # Generate line
    p = y.shape[0]
    idx = np.random.randint(0, p, size=1)[0]
    u = np.zeros(shape=p)
    u[idx] = 1.0

    # Project and split
    proj_y = np.dot(u, y)
    return mse(proj_y)


def projection_random(y, sparse=0.1):
    """
    Project onto a randomly generated sparse oblique line and split.
    """

    # Generate line
    p = y.shape[0]
    u = np.zeros(shape=p)

    max_nonzero = max(1, np.floor(p * sparse))
    num_nonzero = np.random.randint(1, max_nonzero)

    for idx in range(num_nonzero):
        u[idx] = np.random.choice([-1, 1])

    u = u / np.linalg.norm(u)
    np.random.shuffle(u)

    # Project and split
    proj_y = np.dot(u, y)
    return mse(proj_y)
