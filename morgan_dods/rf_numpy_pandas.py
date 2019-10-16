import numpy as np
import pandas as pd
import random
import math

# Test with Iris and Sonar dataset
# need to make pandas/numpy version of RF


def cross_validation_split(data, n_folds):
    cols = list(data.columns)
    fold_size = int(len(data) / n_folds) # find # of samples per fold
    data_split = [] 
    data_copy = data.copy()
    for i in range(n_folds):
        fold = pd.DataFrame(columns = cols)
        while len(fold) < fold_size: 
            # add samples to fold list until size reached
            index = random.randrange(len(data_copy.index))
            fold = fold.append(data_copy.loc[data_copy.index[index]])
            data_copy = data_copy.drop(data_copy.index[index])
        # then add to list of folds
        data_split.append(fold)
    return data_split # return list of fold dfs
 
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    # for each idx in y_true and y_pred, add to num correct
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0 # calculate accuracy
 
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(data, algorithm, n_folds, *args):
    # split data into n folds saved in folds (a list of folds (each with samples))
    folds = cross_validation_split(data, n_folds)
    scores = []
    for fold_idx in range(len(folds)): # for every fold
        train_set_ = list(folds) #copy whole folds list of lists
        del train_set_[fold_idx] # remove fold of interest
        #train_set = sum(train_set, []) # add an empty list to the end
        cols = list(data.columns)
        train_set = pd.DataFrame(columns = cols)
        for i in train_set_:
            train_set = train_set.append(i)
        test_set = pd.DataFrame(columns = cols)
        # for each sample in the fold
        for index, row in folds[fold_idx].iterrows():
            row_copy = row.copy()
            row_copy.pop(cols[-1])
            test_set = test_set.append(row_copy) # copy sample and append it to test set
            #row_copy[-1] = None # remove label from sample
        predicted = algorithm(train_set.reset_index(), test_set.reset_index(), *args) #predict based on clf w/ train and test set
        #return predicted
        actual = [row[cols[-1]] for index, row in folds[fold_idx].iterrows()] # generate y_true
        accuracy = accuracy_metric(actual, predicted) # compute accuracy score
        #return accuracy
        scores.append(accuracy) # add to a list of scores for every split
    return scores # return scores for each split
 
# Split a dataset based on an attribute and an attribute value
def test_split(feat, value, data):
    left = data[data[feat] < value]
    right = data[data[feat] >= value]
    #for row in dataset: # for each sample 
    #    if row[index] < value: # split samples based on a single feature's value
    #        left.append(row)
    #    else:
    #        right.append(row)
    return left.reindex(), right.reindex()
 
# Calculate the Gini index for a split dataset
def gini_index(groups, classes, label_name):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups])) 
    # sum list of lengths of groups (2 way split based on one feature value)
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            #p = [row[-1] for row in group].count(class_val) / size #count all appearances of class val in group 
            p = len(group[group[label_name] == class_val]) / size                              # divide by length of group
            score += p * p # add square of this to score
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances) 
    return gini
 
# Select the best split point for a dataset
def get_split(data, n_feats):
    cols = list(data.columns)
    # obtain unique class labels
    class_values = np.unique(data[cols[-1]])
    # best index, value, score, and groups
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    #empty list of features
    features = []
    columns = list(data.columns)
    # select random feature indices until n_feats reached
    while len(features) < n_feats:
        index = random.randrange(1,data.shape[1]-1)
        if columns[index] not in features:
            features.append(columns[index])
    # for each feature idx and sample
    for feat in features:
        for index, row in data.iterrows():
            # groups is a 2 way split based on the selected index and the value of that feature in that sample
            groups = test_split(feat, row[feat], data)
            gini = gini_index(groups, class_values, cols[-1]) # calculate gini score for each group (low is good)
            if gini < b_score: # get the best split over all selected features and samples
                b_feat, b_value, b_score, b_groups = feat, row[feat], gini, groups
    return {'index':b_feat, 'value':b_value, 'groups':b_groups}
 
# Create a terminal node value
def to_terminal(group):
    cols = list(group.columns)
    #outcomes = row['species'].value_counts().max() # list of y_true for L and R groups
    return group[cols[-1]].value_counts().idxmax() # which class appears most in this group?
 
# Create child splits for a node or make terminal
def split(node, max_depth, min_size, n_features, depth):
    left, right = node['groups'] # copy L and R groups for node
    del(node['groups']) # delete groups from the node
    # check for a no split
    if len(left) < 1 or len(right) < 1: # what is not a list?? no split?
        node['left'] = node['right'] = to_terminal(left.append(right)) #means this node is a leaf node
        return
    # check for max depth
    if depth >= max_depth: # if at max depth make left and right terminal nodes
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size: # check if enough samples in left node to split
        node['left'] = to_terminal(left) # if not, this is a leaf node
    else:
        node['left'] = get_split(left, n_features) # otherwise, get the best split again
        split(node['left'], max_depth, min_size, n_features, depth+1) # run this function again
    # process right child
    if len(right) <= min_size: # same as left
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth+1)
 
# Build a decision tree
def build_tree(train, max_depth, min_size, n_features):
    root = get_split(train, n_features) # get split with lowest gini score for training set
    split(root, max_depth, min_size, n_features, 1) # recursively split until no split, max_depth, or min_size reached
    return root # split alters the tree, which is a dictionary of dictionaries

# Make a prediction with a decision tree
def predict(node, row):
    # node['left'] and node['right'] is either a dict or an int depending on whether terminal
    if row[node['index']] < node['value']: # determine whether sample falls on left or right side of this node
        if isinstance(node['left'], dict):
            return predict(node['left'], row) # continue down the tree unless a leaf node is passed
        else:
            return node['left'] # if a leaf node is reached return the class value
    else: # same for right side
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']
        
# Create a random subsample from the dataset with replacement
def subsample(data, ratio):
    cols = list(data.columns)
    sample = pd.DataFrame(columns = cols) # empty sample
    n_sample = round(len(data) * ratio) # number of samples based on ratio of dataset to sample
    while len(sample) < n_sample: # get n samples by randomly choosing indices
        index = random.randrange(len(data.index))
        sample = sample.append(data.iloc[data.index[index]])
    return sample # list of samples (not indices)

# Make a prediction with a list of bagged trees
def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees] # list of predictions for a sample for all trees
    return max(set(predictions), key=predictions.count) # most common prediction

# Random Forest Algorithm
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    trees = [] # empty list of trees
    for i in range(n_trees): # generate n tree dicts and put them into tree list
        sample = subsample(train, sample_size) # train on random samples from training set
        tree = build_tree(sample, max_depth, min_size, n_features) #build tree using max depth, min size, and n features
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for index, row in test.iterrows()] #generate list of predictions for test set
    return(predictions) # return this list of test predictions (Why isn't this a separate function??)



'''
# Split a dataset into k folds
def cross_validation_split(data, n_folds):
    data_split = []
    data_copy = list(data)
    fold_size = int(len(dataset) / n_folds) # find # of samples per fold
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size: 
            # add samples to fold list until size reached
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold) # then add to list of folds
    return dataset_split # return list of fold lists
 
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    # for each idx in y_true and y_pred, add to num correct
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0 # calculate accuracy
 
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    # split data into n folds saved in folds (a list of folds (each with samples))
    folds = cross_validation_split(dataset, n_folds)
    
    scores = list()
    for fold in folds: # for every fold
        train_set = list(folds) #copy whole folds list of lists
        train_set.remove(fold) # remove fold of interest
        train_set = sum(train_set, []) # add an empty list to the end
        test_set = list() 
        # for each sample in the fold
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy) # copy sample and append it to test set
            row_copy[-1] = None # remove label from sample
        predicted = algorithm(train_set, test_set, *args) #predict based on clf w/ train and test set
        actual = [row[-1] for row in fold] # generate y_true
        accuracy = accuracy_metric(actual, predicted) # compute accuracy score
        scores.append(accuracy) # add to a list of scores for every split
    return scores # return scores for each split
 
# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset: # for each sample 
        if row[index] < value: # split samples based on a single feature's value
            left.append(row)
        else:
            right.append(row)
    return left, right
 
# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups])) 
    # sum list of lengths of groups (2 way split based on one feature value)
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size #count all appearances of class val in group 
                                                                    # divide by length of group
            score += p * p # add square of this to score
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances) 
    return gini
 
# Select the best split point for a dataset
def get_split(dataset, n_features):
    # obtain unique class labels
    class_values = list(set(row[-1] for row in dataset))
    # best index, value, score, and groups
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    #empty list of features
    features = list()
    # select random feature indices until n_feats reached
    while len(features) < n_features:
        index = randrange(len(dataset[0])-1)
        if index not in features:
            features.append(index)
    # for each feature idx and sample
    for index in features:
        for row in dataset:
            # groups is a 2 way split based on the selected index and the value of that feature in that sample
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values) # calculate gini score for each group (low is good)
            if gini < b_score: # get the best split over all selected features and samples
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}
 
# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group] # list of y_true for L and R groups
    return max(set(outcomes), key=outcomes.count) # which class appears most in this group?
 
# Create child splits for a node or make terminal
def split(node, max_depth, min_size, n_features, depth):
    left, right = node['groups'] # copy L and R groups for node
    del(node['groups']) # delete groups from the node
    # check for a no split
    if not left or not right: # what is not a list?? no split?
        node['left'] = node['right'] = to_terminal(left + right) #means this node is a leaf node
        return
    # check for max depth
    if depth >= max_depth: # if at max depth make left and right terminal nodes
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size: # check if enough samples in left node to split
        node['left'] = to_terminal(left) # if not, this is a leaf node
    else:
        node['left'] = get_split(left, n_features) # otherwise, get the best split again
        split(node['left'], max_depth, min_size, n_features, depth+1) # run this function again
    # process right child
    if len(right) <= min_size: # same as left
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth+1)
 
# Build a decision tree
def build_tree(train, max_depth, min_size, n_features):
    root = get_split(train, n_features) # get split with lowest gini score for training set
    split(root, max_depth, min_size, n_features, 1) # recursively split until no split, max_depth, or min_size reached
    return root # split alters the tree, which is a dictionary of dictionaries

# Make a prediction with a decision tree
def predict(node, row):
    # node['left'] and node['right'] is either a dict or an int depending on whether terminal
    if row[node['index']] < node['value']: # determine whether sample falls on left or right side of this node
        if isinstance(node['left'], dict):
            return predict(node['left'], row) # continue down the tree unless a leaf node is passed
        else:
            return node['left'] # if a leaf node is reached return the class value
    else: # same for right side
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']
        
# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):
    sample = list() # empty sample
    n_sample = round(len(dataset) * ratio) # number of samples based on ratio of dataset to sample
    while len(sample) < n_sample: # get n samples by randomly choosing indices
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample # list of samples (not indices)

# Make a prediction with a list of bagged trees
def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees] # list of predictions for a sample for all trees
    return max(set(predictions), key=predictions.count) # most common prediction

# Random Forest Algorithm
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    trees = list() # empty list of trees
    for i in range(n_trees): # generate n tree dicts and put them into tree list
        sample = subsample(train, sample_size) # train on random samples from training set
        tree = build_tree(sample, max_depth, min_size, n_features) #build tree using max depth, min size, and n features
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test] #generate list of predictions for test set
    return(predictions) # return this list of test predictions (Why isn't this a separate function??)


'''