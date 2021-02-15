'''
Functions for creating, trainin and evaluating models.
'''
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

from create_folds import get_kfold_stats
    
def get_pred_data(preds, data, names = ['xgb', 'lgbm', 'rfc']):
    '''
    Using output of get_wrong_preds, gets dataframe representing how
    models performed on the examples which were incorrectly classified.
    '''
    def loc(tup):
        return tup[0]
    pred_data = data.copy()
    for num in range(len(names)):
        preds[num].sort(key=loc)
        pred_data[names[num]] = np.array(preds[num])[:, 1]
    return pred_data


def test_lgbm(lrs, n_leav, n_ests, m_depths, boost_type, seed_num):
    '''
    Test accuracy of lgbm for each value in lists per each perameter.
    '''
    accs = []
    params = []
    for num in n_leav:
        for lr in lrs:
            for n_est in n_ests:
                for depth in m_depths:
                    for boost in boost_type:
                        seed_acc = 0
                        for seed in np.random.randint(1, 100, size=seed_num):
                            model = LGBMClassifier(boosting_type=boost,max_depth=depth,
                                                   num_leaves=num, learning_rate=lr,
                                                   n_estimators=n_est)
                            acc, _, p = get_kfold_stats(X, y, 5, seed_num, models=[model])
                            seed_acc += acc[0] / 15
                        accs.append(seed_acc)
                        params.append([boost, lr, num, n_est, depth,])
    best_acc = max(accs)
    best_params = params[accs.index(max(accs))]
    return (accs, params, best_acc, best_params)


def model_acc(model, X_test, y_test):
    '''
    Returns accuracy and predictions of a model.
    '''
    preds = model.predict(X_test)
    return(accuracy_score(y_test, preds), preds)


def parameter_generator(parameters, names):
    '''
    Given a list of lists containing parameters, and a list of names
    yields every combination of the parameters in the lists.
    '''
    indices = [0 for x in names]
    max_vals = [len(x) for x in parameters]
    while indices[0] < len(parameters[0]):
        params = {}
        for i, param in enumerate(parameters):
            params[names[i]] = param[indices[i]]
        indices = increment_index(indices, max_vals)
        yield params

          
def increment_index(indices, max_vals):
    '''
    Recursively increments the indices of several lists so that
    every combination of elements of those lists can be seen.
    
    Arguments -------
    indices = list of indices for lists
    max_vals = length of each list
    '''
    indices[-1] += 1
    if indices[-1] > max_vals[-1] - 1 and len(indices) > 1:
        indices[-1] = 0
        indices[0:-1] = increment_index(indices[0:-1], max_vals[0:-1])
    return indices