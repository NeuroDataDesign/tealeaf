import numpy as np


class GrossErrorModel:
    """
    Gross Error Model for sampling outlier distribution.

    Parameters
    ----------
    n_features : int
        Number of input features
    n_targets : int
        Number if output targets to predict
    u_1 : float (default=0)
        Mean of the gross population
    u_2 : float (default=5)
        Mean of the true population
    """

    def __init__(self, n_features, n_targets, u_1=0, u_2=5):

        self.u_1 = np.repeat(u_1, n_features + n_targets)
        self.u_2 = np.repeat(u_2, n_features + n_targets)

        self.n_features = n_features
        self.n_targets = n_targets
        self.cov_1 = self._build_random_covariance_matrix(n_features, n_targets)
        self.cov_2 = self._build_random_covariance_matrix(n_features, n_targets)

    def _build_random_covariance_matrix(self, n_features, n_targets):

        A_1 = np.identity(n=n_features)
        A_2 = np.random.uniform(size=(n_targets, n_targets))
        A_2 = (A_2 + A_2.T) / 2.0
        np.fill_diagonal(A_2, 1)
        B = np.random.uniform(size=(n_targets, n_features))

        cov = np.hstack([np.vstack([A_1, B]), np.vstack([B.T, A_2])])

        return cov

    def _sample(self, n_samples, population):

        if population == "true":
            samples = np.random.multivariate_normal(
                mean=self.u_1, cov=self.cov_1, size=n_samples
            )
        elif population == "gross":
            samples = np.random.multivariate_normal(
                mean=self.u_2, cov=self.cov_1, size=n_samples
            )
        else:
            raise ValueError(f"Unknown model type {population}")

        X = samples[:, : self.n_features]
        y = samples[:, self.n_features :]
        return X, y

    def sample(self, n_samples, epsilon):

        n_gross = np.random.binomial(n=n_samples, p=epsilon)
        n_true = n_samples - n_gross

        X_gross, y_gross = self._sample(n_gross, population="gross")
        X_true, y_true = self._sample(n_true, population="true")
        X = np.concatenate((X_gross, X_true), axis=0)
        y = np.concatenate((y_gross, y_true), axis=0)
        model_label = np.concatenate((np.repeat(0, n_gross), np.repeat(1, n_true)))

        return X, y, model_label
