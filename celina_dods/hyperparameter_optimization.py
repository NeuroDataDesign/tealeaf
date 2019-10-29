from sklearn.model_selection import RandomizedSearchCV


def hyperparameter_optimization(X,y,*argv):
    
    clf_best_params = {}
    for clf, params in argv:

        # run randomized search     
        n_iter_search = 10
        random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                           n_iter=n_iter_search, cv=10, iid=False)
        random_search.fit(X, y)       
        clf_best_params[clf] = random_search.best_params_  
    return clf_best_params
