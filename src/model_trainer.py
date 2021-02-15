'''
Class for making training and optimization of models simpler
'''
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from model_training import parameter_generator
from model_training import model_acc, increment_index
from create_folds import kfold, get_kfold_stats

class ModelTrainer():
    '''
    Class for training, testing and optimizing machine learning algorithms.
    '''

    def __init__(self, models, X, y, names):
        '''
        Initialize training data, and models to be trained.

        Arguments -------
        models: list of uninitiated ML algorithm objects, i.e 
        [LinearRegression, LogisticRegression] not [LinearRegression(),
        LogisticRegression()]
        X: X dataset
        y: y dataset
        names: string to represent each model in models
        '''
        self._models = models
        self._X = X
        self._y = y
        self._names = names
    
    def ttt_models(self, test_size=0.2, random_state=33, model_params=None):
        '''
        Train model on data split using train_test_split with parameters shown

        Arguments -------
        test_size:(optional) train_test_split argument for size of test dataset
        random_state: (optional) random state of train_test_split
        model_params: (optional) list of dics full of params per model in
        self._models each model must have a dict.
        '''
        X_train, X_test, y_train, y_test = train_test_split(self._X, self._y,
         test_size=test_size, random_state=random_state)
        accs = []
        predictions = []
        for i, model_obj in enumerate(self._models):
            model = model_obj()
            if model_params:
                model = model_obj(**model_params[i])
            model.fit(X_train, y_train)
            acc, preds = model_acc(model, X_test, y_test)
            predictions.append(preds)
            accs.append(acc)
            print('' + self._names[i] + ': ' + str(acc))
        return accs, preds, X_test, y_test
    

    def kfold_models(self, k, seed=33, model_params=None):
        '''
        Train model on data split using kfolds cross validation.
        '''
        accs = [0 for model in self._models]
        index_pred = [[] for model in self._models]
        iterator = kfold(self._X, self._y, k, seed=seed)
        for X_train, y_train, X_test, y_test, train_index,test_index in iterator:
            for i, model_obj in enumerate(self._models):
                model = model_obj()
                if model_params:
                    model = model_obj(**model_params[i])
                model.fit(X_train, y_train)
                acc, preds = model_acc(model, X_test, y_test)
                accs[i] += acc / k
                index_pred[i] += zip(test_index, preds)
                
        return accs, index_pred
    
    def model_optimizer(self, parameters, param_names, num_seeds=15,
     verbose=False):
        '''
        Tries all combinations of all parametes for each model, returns
        dictionary containing results of each run per each model.
        Arguments -------
        parameters: list of 1 list per model where each inner list contains
        lists of parameter values.
        param_names: list of 1 list per model where each inner list contains
        the names of parameters being optimized in the same order as in
        parameters.
        num_seeds: number of random seeds to run 5 kfold test on per parameter
        combination
        verbose: if true prints each model name, parameter combination, and
        accuracy
        '''
        final = {}
        for name in self._names:
            final[name + ' accs'] = []
            final[name + ' params'] = []

        for i, model_obj in enumerate(self._models):
            if verbose:
                print(self._names[i])
                print('--------------------')
            accs = []
            params = []
            for param in parameter_generator(parameters[i], param_names[i]):
                model = model_obj(**param)
                seed_acc = 0
                for seed in np.random.randint(1, 900, size=num_seeds):
                    acc, _, p = get_kfold_stats(self._X, self._y, 5, seed, models=[model])
                    seed_acc += acc[0] / 15
                accs.append(seed_acc)
                params.append(param)
                if verbose:
                    print(param, seed_acc)
            final[self._names[i] + ' accs'].append(accs)
            final[self._names[i] + ' params'].append(params)
        return final