import pickle as pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from clean_data import clean_data
from sklearn.model_selection import train_test_split
from sklearn import metrics

class FraudClassifier(object):
    '''A fraud classifier model:
        - Fit a random forest model to the resulting features.
    '''

    def __init__(self):
        # self._classifier = RandomForestClassifier(max_features=None, n_jobs=-1, n_estimators=20, class_weight='balanced')
        self._classifier = Pipeline([('scaler', StandardScaler()),
            ('rfc', RandomForestClassifier(max_features=None, n_jobs=-1, n_estimators=20, class_weight='balanced')),
        ])

    def fit(self, X, y):
        '''Fit a fraud classifier model.

        Parameters
        ----------
        X: A numpy array of the feature space
        y: A numpy array of target values

        Returns
        -------
        self: The fitted model object.
        '''
        self._classifier.fit(X, y)
        return self

    def predict_proba(self, X):
        '''Make probability predictions on new data.'''
        predicted_probs = self._classifier.predict_proba(X)
        return predicted_probs

    def predict(self, X):
        '''Make predictions on new data.'''
        predicted = self._classifier.predict(X)
        return predicted

    def score(self, y_true, y_predict):
        '''Return a classification recall score on new data.'''
        recall = recall_score(y_true, y_predict)
        report = metrics.classification_report(y, fc.predict(X))
        return report


def get_data(filename=None):
    '''Load raw data from a file and return training data and responses.

    Parameters
    ----------
    filename: The path to a json file containing the raw data and response.

    Returns
    -------
    X: A numpy array containing the independent data used for training.
    y: A numpy array containing labels, used for model response.
    '''
    df = pd.DataFrame()
    if filename == None:
        df = pd.read_csv('data/clean_data.csv')
        del df['Unnamed: 0']
    else:
        df = pd.read_json(filename)
        df = clean_data(df, save=True)

    # These columns are only used in nlp
    del df['description']
    del df['org_desc']

    y = df.pop('fraud_target')
    X = df.values
    return X, y


if __name__ == '__main__':
    # X, y = get_data('data/data.json')
    X, y = get_data()
    # X_scaled, scaler = scale_data(X)
    fc = FraudClassifier()
    fc.fit(X, y)

    with open('models/model.pkl', 'wb') as f:
        pickle.dump(fc, f)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
    fc = FraudClassifier()
    fc.fit(X_train, y_train)
    predicted = fc.predict(X_test)
    print(metrics.classification_report(y_test, predicted))
