'''
Cross Validation Functions
'''
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np


def kfold(X, y, k=5, stratify=False, shuffle=False, seed=33):
    """K-Folds cross validation iterator.

    Parameters
    ----------
    k : int, default 5
    stratify : bool, default False
    shuffle : bool, default True
    seed : int, default 33

    Yields
    -------
    X_train, y_train, X_test, y_test, train_index, test_index
    """
    if stratify:
        kf = StratifiedKFold(n_splits=k, random_state=seed, shuffle=shuffle)
    else:
        kf = KFold(n_splits=k, random_state=seed, shuffle=shuffle)
    
    data = np.array(X)
    target = np.array(y)
    for train_index, test_index in kf.split(X, y):
        X_train, y_train = data[train_index], target[train_index]
        X_test, y_test = data[test_index], target[test_index]
        yield X_train, y_train, X_test, y_test, train_index, test_index


def get_kfold_stats(X, y, nsplits, models, seed=33):
    '''
    Train models on X, y using kfold cross validation with nsplits. Models 
    defaults to XGBoost, LightGB, and RandomForestClassifier.
    Returns average accuracy of each model and the incorrect predictions
    and indexes of each incorrectly predicted point.

    Arguments -----
    X features / training variables
    y target / training labels
    nsplits (optional) default: 5 number of splits to use in kfold algorithm
    models (optional) list of ml models to train
    '''
    xlr_accs = [0 for model in models]
    xlr_index_pred = [[] for model in models]
    avg_feature_importance = [np.zeros(X.shape[1]) for model in models]
    for X_train, y_train, X_test, y_test, train_index, test_index in kfold(X, y, nsplits, seed=seed):
        for model in models:
            model.fit(X_train, y_train)
            
        for i, model in enumerate(models):
            acc, preds = model_acc(model, X_test, y_test)
            xlr_accs[i] += acc / nsplits
            xlr_index_pred[i] += zip(test_index, preds)
        try:
            avg_feature_importance[i] += model.feature_importances_
        except:
            pass
            
    return xlr_accs, xlr_index_pred, [x / nsplits for x in avg_feature_importance]


def model_acc(model, X_test, y_test):
    '''
    Returns accuracy and predictions of a model.
    '''
    preds = model.predict(X_test)
    return(accuracy_score(y_test, preds), preds)



