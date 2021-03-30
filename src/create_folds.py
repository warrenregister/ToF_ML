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