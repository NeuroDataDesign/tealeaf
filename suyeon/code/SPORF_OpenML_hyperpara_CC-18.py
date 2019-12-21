# OpenML CC-18

## Load in datasets

import openml
import sklearn
from rerf.rerfClassifier import rerfClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint as sp_randint
import math
from math import log

import warnings
warnings.filterwarnings('ignore')

benchmark_suite = openml.study.get_suite('OpenML-CC18')  # obtain the benchmark suite

## Hyperparameter optimization function
def hyperparameter_optimization_grid(X, y, *argv):
    """
    Given a classifier and a dictionary of hyperparameters, find optimal hyperparameters using GridSearchCV.
    Parameters
    ----------
    X : numpy.ndarray
        Input data, shape (n_samples, n_features)
    y : numpy.ndarray
        Output data, shape (n_samples, n_outputs)
    *argv : list of tuples (classifier, hyperparameters)
        List of (classifier, hyperparameters) tuples:
        classifier : sklearn-compliant classifier
            For example sklearn.ensemble.RandomForestRegressor, rerf.rerfClassifier, etc
        hyperparameters : dictionary of hyperparameter ranges
            See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html.
    Returns
    -------
    clf_best_params : dictionary
        Dictionary of best hyperparameters
    """

    clf_best_params = {}

    # Iterate over all (classifier, hyperparameters) pairs
    for clf, params in argv:

        # Run grid search
        grid_search = GridSearchCV(
            clf, param_grid=params, cv=10, iid=False
        )
        grid_search.fit(X, y)

        # Save results
        clf_best_params[clf] = grid_search.best_params_

    return clf_best_params

## Find optimized hyperparameters

dimen_CC18 = []
best_params = []

for task_id in benchmark_suite.tasks[61:62]:  # iterate over all tasks
    # try:
        # print(estimator.get_params().keys())

        f = open("SPORF_accuracies_CC-18_hyperpara.txt","a")
        task = openml.tasks.get_task(task_id)  # download the OpenML task
        X_CC18, y_CC18 = task.get_X_and_y()  # get the data
        dimen_CC18.append(np.shape(X_CC18))
        n_features = np.shape(X_CC18)[1]
        n_samples = np.shape(X_CC18)[0]

        # build a classifier
        clf = rerfClassifier(n_estimators=100)
        
        #specify max_depth and min_sample_splits ranges
        max_depth_array = (np.unique(np.round((np.arange(2,math.log(n_samples),
                            (math.log(n_samples)-2)/10))))).astype(int)
        max_depth_range_rerf = np.append(max_depth_array, None)

        min_sample_splits_range = (np.unique(np.round((np.arange(2,math.log(n_samples),
                                    (math.log(n_samples)-2)/10))))).astype(int)

        # specify parameters and distributions to sample from
        param_dist = {rerfClassifier(feature_combinations=1.5, image_height=None, image_width=None,
                           max_depth=None, max_features='auto', min_samples_split=1,
                           n_estimators=100, n_jobs=None, oob_score=False,
                           patch_height_max=None, patch_height_min=1, patch_width_max=None,
                           patch_width_min=1, projection_matrix='RerF', random_state=None): 
                                {'n_estimators': 350, 
                                'min_samples_split': 1, 
                                'max_features': 36, 
                                'max_depth': 1536, 
                                'feature_combinations': 2}, 
                    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                            max_depth=None, max_features='auto', max_leaf_nodes=None,
                            min_impurity_decrease=0.0, min_impurity_split=None,
                            min_samples_leaf=1, min_samples_split=2,
                            min_weight_fraction_leaf=0.0, n_estimators=100,
                            n_jobs=None, oob_score=False, random_state=None,
                            verbose=0, warm_start=False): 
                                {'n_estimators': 500, 
                                 'min_samples_split': 2, 
                                 'max_features': None, 
                                 'max_depth': 961}
                    }    

        keys, values = zip(*param_dist.items())

        #train test split
        X_train, X_test, y_train, y_test = train_test_split(X_CC18, y_CC18, test_size=0.33, random_state=42)

        rerf_opti = rerfClassifier(**values[0])
        rerf_opti.fit(X_train, y_train)
        rerf_pred_opti = rerf_opti.predict(X_test)
        rerf_accuracy_opti = metrics.accuracy_score(y_test, rerf_pred_opti)
        print(rerf_accuracy_opti)

        print(task_id)
        print('Data set: %s: ' % (task.get_dataset().name))
        # default = "rerfClassifier(feature_combinations=1.5, image_height=None, image_width=None,max_depth=None, max_features='auto', min_samples_split=1,n_estimators=100, n_jobs=None, oob_score=False,patch_height_max=None, patch_height_min=1, patch_width_max=None,patch_width_min=1, projection_matrix='RerF', random_state=None)"
        # print(clf_best_params[default]["feature_combinations"])
        # print(clf_best_params[default]["max_depth"])
        # print(clf_best_params[default]["max_features"])
        # print(clf_best_params[default]["min_samples_split"])
        # print(clf_best_params[default]["n_estimators"])
        print('Time: '+ str(datetime.now() - startTime))
        # f.write('%i,%s,%s,%f,%f,%f,%f,%f\n' % (task_id,task.get_dataset().name,str(datetime.now() - startTime),clf_best_params["feature_combinations"],clf_best_params["max_depth"],clf_best_params["max_features"],clf_best_params["min_samples_split"],clf_best_params["n_estimators"]))
        f.close()
    # except:
    #     print('Error in OpenML CC-18 dataset ' + str(task_id))


