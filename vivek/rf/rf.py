from collections import namedtuple

import numpy as np

from split import mae, mse, projection_axis, projection_random

# Named tuple for potential splits
Group = namedtuple("Group", "X y")


def partition(X, y, feature, split):
    """
    Partition data based on a given split
    """

    indices_1 = []
    indices_2 = []

    for idx, row in enumerate(X):
        if row[feature] > split:
            indices_1.append(idx)
        else:
            indices_2.append(idx)

    group_1 = Group(X[indices_1], y[indices_1])
    group_2 = Group(X[indices_2], y[indices_2])

    return group_1, group_2


def find_best_partition(X, y, criteria, n_features, min_leaf_size):
    """
    Find the best split at a given node
    """

    # Choose random features
    feature_idx = np.arange(0, X.shape[1])
    chosen_features = np.random.choice(feature_idx, size=n_features, replace=False)

    # Track the best score
    best_score = 0
    found = False

    # Iterate over possible (feature, split) combinations
    for i in chosen_features:
        possible_splits = np.unique([row[i] for row in X])
        for j in possible_splits:

            group_1, group_2 = partition(X, y, feature=i, split=j)

            if len(group_1.X) < min_leaf_size or len(group_2.X) < min_leaf_size:
                continue

            score = (
                criteria(y) * len(y)
                - criteria(group_1.y) * len(group_1.y)
                - criteria(group_2.y) * len(group_2.y)
            )

            if score > best_score:
                found = True
                best_score = score
                best_group_1 = group_1
                best_group_2 = group_2
                best_split = j
                best_feature = i

    if found is False:
        best_split = 1
        best_feature = 1
        best_group_1 = Group(X, y)
        best_group_2 = Group(np.array([]), np.array([]))

    return best_group_1, best_group_2, best_split, best_feature, best_score


class RandomForestNode:
    def __init__(self, X, y, criteria, max_depth, n_features, min_leaf_size, depth=0):
        self.X = X
        self.y = y
        self.criteria = criteria
        self.terminal = False
        self.max_depth = max_depth
        self.depth = depth
        self.n_features = n_features
        self.min_leaf_size = min_leaf_size

    def _reached_stop(self):
        """
        Determine if node has reached stop criteria
        """
        if (self.depth == self.max_depth) or (len(self.X) < self.min_leaf_size):
            return True
        else:
            return False

    def _get_prediction(self):
        """
        Prediction is the mean of the response variables
        """
        return np.mean(self.y, axis=0)

    def split(self):
        """
        Split node and create child nodes
        """
        if self._reached_stop():
            self.terminal = True
            self.prediction = self._get_prediction()
        else:
            group_1, group_2, self.split, self.feature, score = find_best_partition(
                self.X, self.y, self.criteria, self.n_features, self.min_leaf_size
            )

            if len(group_1.X) == 0 or len(group_2.X) == 0:
                self.terminal = True
                self.prediction = self._get_prediction()
            else:
                self.left = RandomForestNode(
                    group_1.X,
                    group_1.y,
                    self.criteria,
                    self.max_depth,
                    self.n_features,
                    self.min_leaf_size,
                    self.depth + 1,
                )
                self.right = RandomForestNode(
                    group_2.X,
                    group_2.y,
                    self.criteria,
                    self.max_depth,
                    self.n_features,
                    self.min_leaf_size,
                    self.depth + 1,
                )
                self.left.split()
                self.right.split()

    def predict(self, test_data):
        if self.terminal:
            return self.prediction
        else:
            if test_data[self.feature] > self.split:
                return self.left.predict(test_data)
            else:
                return self.right.predict(test_data)


class RandomForest:
    def __init__(
        self, criteria, max_depth, n_features, n_trees, n_bagging, min_leaf_size=5
    ):
        self.criteria = self._get_criteria(criteria)
        self.max_depth = max_depth
        self.n_features = n_features
        self.n_trees = n_trees
        self.n_bagging = n_bagging
        self.min_leaf_size = min_leaf_size

    def _get_criteria(self, criteria):
        if criteria == "mae":
            return mae
        elif criteria == "mse":
            return mse
        elif criteria == "projection_axis":
            return projection_axis
        elif criteria == "projection_random":
            return projection_random
        else:
            raise (ValueError, "Unknown split criteria")

    def _build_tree(self, X, y):
        root = RandomForestNode(
            X, y, self.criteria, self.max_depth, self.n_features, self.min_leaf_size
        )
        root.split()
        return root

    def fit(self, X, y):
        """
        Train an ensemble of trees.

        For each tree, train on a randomly sampled subset of the data (without replacement).
        """

        assert X.ndim == y.ndim == 2, "X and y must be shape (n, p) and (n, q)"

        self.forest = []

        for _ in range(self.n_trees):

            # Randomly sample the data
            idxs = np.arange(0, X.shape[0])
            chosen_input = np.random.choice(idxs, size=self.n_bagging, replace=False)
            bag_x = X[chosen_input]
            bag_y = y[chosen_input]

            # Train a tree and add to the forest
            tree = self._build_tree(bag_x, bag_y)
            self.forest.append(tree)

        return self

    def predict(self, X, method="mean"):
        """
        Return predictions for every element in X.

        Parameters
        ==========
        X : array of shape (n_samples, n_features)
            Input data
        method : string ("mean" (default), "full", "quantile")
            "mean" : return average of predictions for each tree
            "full" : return all predictions from each tree

        Return
        ======
        yhat : array of shape (n_samples, n_predictors)
            Predicted outputs
        """

        yhat = []

        for xi in X:
            yi = np.array([tree.predict(xi) for tree in self.forest])

            if method == "mean":
                yi = np.mean(yi, axis=0)
            elif method == "full":
                yi = yi
            elif method == "quantile":
                # TODO: implement quantile method
                raise NotImplementedError("Quantile method not implemented")
            else:
                raise ValueError(f"Undefined method: {method}")

            yhat.append(yi)

        return yhat
