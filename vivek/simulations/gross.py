import numpy as np
from sklearn.datasets import make_spd_matrix


class GrossErrorModel:
    """
    Gross Error Model for sampling outlier distribution.

    Parameters
    ----------
    n_features : int
        Number of input features
    n_targets : int
        Number if output targets to predict
    """

    def __init__(self, n_features, n_targets, n_informative, transform=np.sin, sigma=5):

        self.n_features = n_features
        self.n_targets = n_targets
        self.n_informative = n_informative

        self.weights = self._make_weight_matrix()
        self.transform = transform

        self.cov_1 = make_spd_matrix(n_dim=n_targets)
        self.cov_2 = self.cov_1 * sigma

    def _make_weight_matrix(self):
        """
        Make a sparse weight matrix.
        """

        weights = np.zeros((self.n_features, self.n_targets))
        weights[: self.n_informative, :] = np.random.uniform(
            low=-10, high=10, size=(self.n_informative, self.n_targets)
        )
        weights = np.random.shuffle(weights)

        return weights

    def sample(self, epsilon, n_iter=25, n_train=30, n_test=1000):
        """
        Create a generator for training data and a heldout validation set.
        """

        # Create sample training data
        train_gen = self._create_training_data(epsilon, n_iter, n_train)

        # Create validation set
        X_test = np.random.uniform(low=-10, high=10, size=(n_test, self.n_features))
        y_test = np.dot(X_test, self.weights)

        return train_gen, X_test, y_test

    def _create_training_data(self, epsilon, n_iter, n_train):
        """
        Create a generator of training data (X_train, y_train).
        """

        for _ in range(n_iter):
            X_train, y_train = self._sample_train(epsilon, n_iter, n_train)
            yield X_train, y_train

    def _sample_train(self, epsilon, n_iter, n_train):
        """
        Sample contaminated training data.
        """

        # Sample uncontanimated data
        X = np.random.uniform(low=-10, high=10, size=(n_train, self.n_features))

        # Index contaminated data
        n_contaminated = int(round(n_train * epsilon))
        contaminated = np.random.choice(range(n_train), size=n_contaminated)
        not_contaminated = np.setdiff1d(range(n_train), contaminated)

        # Contaminate a subset of the data
        y = np.zeros((n_train, self.n_targets))
        y[not_contaminated] = np.dot(X, self.weights) + np.random.normal(0, self.cov_1)
        y[contaminated] = self.transform(
            np.dot(X, self.weights) + np.random.normal(0, self.cov_2)
        )

        return X, y
