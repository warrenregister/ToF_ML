from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from joblib import load, dump
from os import listdir
import numpy as np


class ToFCalibrationDetector:
    """
    Object for training models to predict the calibration of a spectrum
    using descriptive statistics, as well as for quickly predicting the
    calibration of new spectra.
    """

    def __init__(self, k=10, model=LGBMClassifier, dire=None):
        """
        Initiate fields of object. Models created using kfold cross validation,
        k specifies both how many folds and how many modles to use. If dir is
        passed models are loaded from given folder assuming that models have
        been saved by this object.

        Arguments -------
        k: number of folds for kfold cross validation
        model: (Optional) type of model to use, default LGBMClassifier
        dir: (Optional) directory in which models were saved previously
        """
        self._k = k
        self._alg = model

        if dir:
            self._models = self._load_models(dire)
            self._trained = True
        else:
            self._models = []
            self._trained = False

    def fit(self, X, y, model_params=None, stratify=False, shuffle=False,
            seed=33, train_acc=False, verbose=False):
        """
        Fits models to the passed in data using kfold cross validation.

        Arguments -------
        X: features
        y: targets / labels
        model_params: dictionary of parameters for model being trained.
        """
        i = 0
        avg_acc = 0
        for X_tr, y_tr, X_te, y_te, tr_index, te_index in self._kfold(X, y,
                                                                      self._k,
                                                                      stratify,
                                                                      shuffle,
                                                                      seed):
            if model_params:
                curr_model = self._alg(**model_params)
            else:
                curr_model = self._alg()
            curr_model.fit(X_tr, y_tr)
            preds = curr_model.predict(X_te)
            score1 = accuracy_score(y_te, preds)
            avg_acc += score1
            if train_acc:
                score2 = accuracy_score(y_tr, curr_model.predict(X_tr))
            self._models.append(curr_model)

            if verbose:
                model_name = 'Model #' + str(i)
                print(model_name + ' test accuracy: ' + str(score1))
                if train_acc:
                    print(model_name + ' train accuracy: ' + str(score2))
            i += 1
        self._trained = True
        return avg_acc / self._k

    @staticmethod
    def _kfold(X, y, k=5, stratify=False, shuffle=False, seed=33):
        """
        K-Folds cross validation iterator.

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

    def predict(self, X, rounded=False):
        """
        Returns prediction for every row in X.

        Arguments -------
        X: dataframe of features for new examples.
        """
        if not self._trained:
            raise Exception('Models are not trained yet!')

        preds = np.zeros(len(X))
        for model in self._models:
            pred = model.predict_proba(X)[:, 1]
            preds += pred
        preds = preds / self._k
        if rounded:
            return np.round(preds)
        return preds

    def score(self, X, y):
        """
        Returns average accuracy score for passed in dataset.

        Arguments -------
        X: dataframe of features
        y: array of corresponding labels
        """
        if not self._trained:
            raise Exception('Models are not trained yet!')

        accuracy = 0
        for model in self._models:
            preds = model.predict(X)
            accuracy += accuracy_score(y, preds)
        return accuracy / self._k

    def get_probabilities(self, X):
        """
        Returns the result of predict_proba on each model, if
        the model has no such function, it will not work

        Arguments -------
        X: dataframe of features for new examples.
        """
        if not self._trained:
            raise Exception('Models are not trained yet!')

        preds = []
        for model in self._models:
            pred = model.predict_proba(X)
            preds.append(pred)
        return preds

    def get_models(self):
        """
        Return all models
        """
        return self._models

    def save_models(self, path):
        """
        Save all models to the passed in directory using joblip.dump.
        """
        for i, model in enumerate(self._models):
            dump(model, path + "model#_" + str(i) + "_errdetector.joblib")

    def _load_models(self, path):
        """
        Load all models saved by this object in the given directory.
        """
        models = []
        for file in listdir(path):
            if file[0:6] == 'model#' and file.split('.')[-1] == 'joblib':
                models.append(load(path + file))
        return models