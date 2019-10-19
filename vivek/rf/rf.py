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
    feature_idx = list(range(0, X.shape[1]))
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


class RFDecisionNode:
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
                self.left = RFDecisionNode(
                    group_1.X,
                    group_1.y,
                    self.criteria,
                    self.max_depth,
                    self.n_features,
                    self.min_leaf_size,
                    self.depth + 1,
                )
                self.right = RFDecisionNode(
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


class RF:
    def __init__(
        self, X, y, criteria, max_depth, n_features, n_trees, n_bagging, min_leaf_size=5
    ):
        assert X.ndim == y.ndim == 2, "X and y must be shape (n, p) and (n, q)"
        self.X = X
        self.y = y
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
            raise ValueError

    def RF_build_tree(self, X, y):
        root = RFDecisionNode(
            X, y, self.criteria, self.max_depth, self.n_features, self.min_leaf_size
        )
        root.split()
        return root

    def create_model(self):
        self.forest = []
        for _ in range(self.n_trees):
            input_idx = list(range(0, len(self.X)))
            chosen_input = np.random.choice(
                input_idx, size=self.n_bagging, replace=False
            )
            bag_x = []
            bag_y = []
            for j in chosen_input:
                bag_x.append(self.X[j])
                bag_y.append(self.y[j])

            temp = self.RF_build_tree(np.array(bag_x), np.array(bag_y))
            self.forest.append(temp)

    def predict(self, test_data):
        temp_result = []
        for tree in self.forest:
            temp_result.append(tree.predict(test_data))
        return temp_result
