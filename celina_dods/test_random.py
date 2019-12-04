import hyperparam_optimization as ho
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import openml
from sklearn import metrics
import numpy as np

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

def test_answer():
    # get some data
    task_id = 146821 #car
    openml.config.apikey = 'c9ea8896542dd998ea42685f14e2bc14'
    benchmark_suite = openml.study.get_suite('OpenML-CC18')
    task = openml.tasks.get_task(task_id)
    X, y = task.get_X_and_y()
    n_features = np.shape(X)[1]
    n_samples = np.shape(X)[0]
    
    # build a classifier
    rf = RandomForestClassifier()
    
    # specify parameters and distributions to sample from
    rf_param_dict = {"max_features": ["sqrt","log2", None]}
    
    #get best param dict
    best_params = ho.hyperparameter_optimization_random(X, y,(rf, rf_param_dict))
    keys, values = zip(*best_params.items())
    
    #train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    clf_opti = RandomForestClassifier(**values[0])
    clf_opti.fit(X_train, y_train)
    y_pred_opti = clf_opti.predict(X_test)
    accuracy_opti = metrics.accuracy_score(y_test, y_pred_opti)
    
    clf_default = RandomForestClassifier()
    clf_default.fit(X_train, y_train)
    y_pred_default = clf_default.predict(X_test)
    accuracy_default = metrics.accuracy_score(y_test, y_pred_default)
    
    errors = []
    if not accuracy_opti >= accuracy_default:
        errors.append("Accuracy of optimized model lower than default.")
    if not len(best_params) == 1:
        errors.append("Parameter dict has wrong number of entries.")
  
    assert not errors, "errors occured:\n{}".format("\n".join(errors))