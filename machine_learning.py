from __future__ import division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from future.utils import iteritems
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix

from sklearn import datasets

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
import theano
from sklearn.neural_network import MLPClassifier


# Machine Learning

# This function can take in any sklearn machine learning algorithm. Examples in the if_name_main below
def run_model(cls, X, y, scale=False, **kwargs):
    # All kwargs are all passed along to SVC object
    model_args = {'model__{}'.format(k): v for k, v in iteritems(kwargs)}

    if scale:
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('model', cls())
        ])
    else:
        model = Pipeline([('model', cls())])

    # Pass all given args to the specified model
    model.set_params(**model_args)
    # Fit the model to the given training data
    model.fit(X, y)

    return model

def run_OLS(X, y):
    ols = sm.OLS(y, X)
    ols.fit()
    return ols

def run_KMeans(X, n_clusters=8, max_iter=300, n_jobs=-1, random_state=None):
    km = KMeans(n_clusters=n_clusters, max_iter=max_iter, n_jobs=n_jobs, random_state=random_state)
    km.fit(X)
    return km

def gridsearch_with_output(estimator, parameter_grid, X_train, y_train):
    '''
        Parameters: estimator: the type of model (e.g. RandomForestRegressor())
                    paramter_grid: dictionary defining the gridsearch parameters
                    X_train: 2d numpy array
                    y_train: 1d numpy array

        Returns:  best parameters and model fit with those parameters
    '''
    model_gridsearch = GridSearchCV(estimator,
                                    parameter_grid,
                                    n_jobs=-1,
                                    verbose=True,
                                    scoring='f1')
    model_gridsearch.fit(X_train, y_train)
    best_params = model_gridsearch.best_params_
    model_best = model_gridsearch.best_estimator_
    print("\nResult of gridsearch:")
    print("{0:<20s} | {1:<8s} | {2}".format("Parameter", "Optimal", "Gridsearch values"))
    print("-" * 55)
    for param, vals in iteritems(parameter_grid):
        print("{0:<20s} | {1:<8s} | {2}".format(str(param),
                                                str(best_params[param]),
                                                str(vals)))
    return best_params, model_best


def get_score(y_true, y_predict):
    return f1_score(y_true, y_predict)
    # return confusion_matrix(y_true, y_predict)

if __name__=='__main__':

    # get data
    df = pd.read_csv('data/clean_data.csv')
    del df['Unnamed: 0']
    del df['acct_type']
    y = df.pop('fraud_target')
    X = df.astype(float).values

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # run all models

    ols = run_OLS(X_train, y_train)
    gnb = run_model(GaussianNB, X_train, y_train, priors=None)
    print('gnb Score: '+ str(get_score(y_test, gnb.predict(X_test))))
    # linr = run_model(LinearRegression, X_train, y_train)
    # print('linr Score: '+ str(get_score(y_test, linr.predict(X_test))))
    # logr = run_model(LogisticRegression, X_train, y_train)
    # print('logr Score: '+ str(get_score(y_test, logr.predict(X_test))))
    rfc = run_model(RandomForestClassifier, X_train, y_train)
    print('rfc Score: '+ str(get_score(y_test, rfc.predict(X_test))))
    lSVC = run_model(LinearSVC, X_train, y_train)
    print('lSVC Score: '+ str(get_score(y_test, lSVC.predict(X_test))))
    svc = run_model(SVC, X_train, y_train, scale=True)
    print('svc Score: '+ str(get_score(y_test, svc.predict(X_test))))
    knn = run_model(KNeighborsClassifier, X_train, y_train, n_neighbors=5, weights='uniform', metric='minkowski', n_jobs=-1)
    print('knn Score: '+ str(get_score(y_test, knn.predict(X_test))))

    gradient_boosting_grid = {'learning_rate': [0.1, 0.05, 0.02, 0.01],
                              'max_depth': [2, 4, 6],
                              'min_samples_leaf': [1, 2, 5, 10],
                              'max_features': [1.0, 0.3, 0.1],
                              'n_estimators': [500],
                              'random_state': [1]}
    gbr_best_params, gbr_best_model = gridsearch_with_output(GradientBoostingClassifier(), gradient_boosting_grid, X_train, y_train)
    '''
    Result of gridsearch:
    Parameter            | Optimal  | Gridsearch values
    -------------------------------------------------------
    learning_rate        | 0.1      | [0.1, 0.05, 0.02, 0.01]
    max_depth            | 6        | [2, 4, 6]
    min_samples_leaf     | 1        | [1, 2, 5, 10]
    max_features         | 0.3      | [1.0, 0.3, 0.1]
    n_estimators         | 500      | [500]
    random_state         | 1        | [1]
    '''

    ada_boosting_grid = {'learning_rate': [1, 0.7, 0.3, 0.1, 0.01],
                              'n_estimators': [500, 1000, 1500, 2000],
                              'random_state': [1]}
    ada_best_params, ada_best_model = gridsearch_with_output(AdaBoostClassifier(), ada_boosting_grid, X_train, y_train)

    rfc_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 5, 8, 12],
              "min_samples_split": [2, 5, 8, 12],
              "min_samples_leaf": [2, 5, 8, 12],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

    rfc_best_params, rfc_best_model = gridsearch_with_output(RandomForestClassifier(), rfc_grid, X_train, y_train)
    '''
    Result of gridsearch:
    Parameter            | Optimal  | Gridsearch values
    -------------------------------------------------------
    max_depth            | None     | [3, None]
    max_features         | 5        | [1, 3, 5, 8, 12]
    min_samples_split    | 5        | [2, 5, 8, 12]
    min_samples_leaf     | 2        | [2, 5, 8, 12]
    bootstrap            | False    | [True, False]
    criterion            | entropy  | ['gini', 'entropy']
    Fitting 3 folds for each of 10 candidates, totalling 30 fits
    '''

    mlp_grid={
        'learning_rate_init': [0.001],
        'activation': ['logistic', 'relu'],
        'solver': ['lbfgs'],
        'hidden_layer_sizes': [(100,1), (10,4), (40,3), (50,2), (60,4)]}

    mlp_best_params, mlp_best_model = gridsearch_with_output(MLPClassifier(), mlp_grid, X_train, y_train)
