"""
Contains helper methods for specific machine learning tasks.
"""
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier


def get_best_params_lgbm(X, y):
    """
    Returns best parameters found for the lgbm classifier model on the dataset
    X, y using sklearn's GridSearchCV.Also returns the score of the best model.

    Arguments -------
    X: Features
    y: labels
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    lgbm = LGBMClassifier()
    parameters = {'learning_rate':[0.01, 0.1, 0.2],
                'max_depth': [-1, 3, 7],
                'num_leaves': [10, 20, 40],
                'boosting_type': ['goss', 'dart']}
    clf_lgbm = GridSearchCV(lgbm, parameters)
    clf_lgbm.fit(X_train, y_train)
    params = clf_lgbm.best_estimator_.get_params()
    return params, clf_lgbm.score(X_test, y_test)


def get_best_params_rfc(X, y):
    """
    Returns best parameters found for the rf classifier model on the dataset
    X, y using sklearn's GridSearchCV.Also returns the score of the best model.

    Arguments -------
    X: Features
    y: labels
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    rfc = RandomForestClassifier()
    parameters = {'criterion':('gini', 'entropy'), 'n_estimators':[75,90, 150],
                  'max_depth':[3,15,31]}
    clf_rfc = GridSearchCV(rfc, parameters)
    clf_rfc.fit(X_train, y_train)
    params = clf_rfc.best_estimator_.get_params()
    return params, clf_rfc.score(X_test, y_test)