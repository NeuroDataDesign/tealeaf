import numpy as np


def mae(y):
    """
    Calcualte the mean absolute error
    """

    y_bar = np.mean(y, axis=0).reshape(-1, 1)
    return np.sum([np.abs(row - y_bar) for row in y])


def mse(y):
    """
    Calculate the mean squared error
    """

    y_bar = np.mean(y, axis=0).reshape(-1, 1)
    return np.sum([np.linalg.norm(row - y_bar) ** 2 for row in y])


def projection_axis(y):

    p = y.shape[0]
    idx = np.random.randint(0, p, size=1)[0]
    u = np.zeros(shape=p)
    u[idx] = 1.0

    proj_y = np.dot(u, y)
    return mse(proj_y)


def projection_random(y, dist=[0.1, 0.8, 0.1]):

    p = y.shape[0]
    u = np.zeros(shape=p)

    for idx, elem in enumerate(u):
        i = np.random.choice([-1, 0, 1], p=dist)
        u[idx] = i
    u = u / np.linalg.norm(u)

    proj_y = np.dot(u, y)
    return mse(proj_y)
