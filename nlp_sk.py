import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from string import punctuation, printable
from sklearn.metrics import mean_squared_error, accuracy_score, recall_score, precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.sklearn

from html.parser import HTMLParser

def clean_data(fname):
    df_all = pd.read_json(fname)
    del df_all['object_id']
    del df_all['has_header']
    del df_all['org_name']
    del df_all['org_desc']
    del df_all['org_facebook']
    del df_all['org_twitter']
    del df_all['name']
    del df_all['sale_duration']
    del df_all['currency']
    del df_all['payee_name']
    del df_all['user_type']

    # Maybes
    del df_all['show_map']
    # del df_all['name_length']
    del df_all['has_analytics']
    del df_all['fb_published']
    del df_all['has_logo']
    # del df_all['channels']
    del df_all['delivery_method']
    del df_all['channels']
    del df_all['user_age']

    # Use later
    del df_all['ticket_types']
    del df_all['previous_payouts']
    # del df_all['description']
    del df_all['email_domain']

    df_all['time_to_payout'] = df_all.apply(lambda x: x['approx_payout_date'] - x['event_end'], axis=1)
    df_all['time_till_event'] = df_all.apply(lambda x: x['event_created'] == x['user_created'], axis=1)
    del df_all['approx_payout_date']
    del df_all['event_end']
    del df_all['event_created']
    del df_all['user_created']
    del df_all['event_published']
    del df_all['event_start']

    del df_all['venue_address']
    del df_all['venue_name']
    del df_all['venue_state']
    del df_all['venue_latitude']
    del df_all['venue_longitude']

    df_all['is_country_same'] = df_all.apply(lambda x: 1 if x['country'] == x['venue_country'] else 0, axis=1)
    del df_all['country']
    del df_all['venue_country']

    df_all['listed'] = df_all.apply(lambda x: 1 if x['listed'] == 'y' else 0, axis=1)

    dummies = pd.get_dummies(df_all['payout_type'], drop_first=True)
    df_all['CHECK'] = dummies['CHECK']
    df_all['ACH'] = dummies['ACH']
    del df_all['payout_type']

    add_fraud_col(df_all)

    return df_all


def add_fraud_col(df):
    '''Add fraud_target Column based on acct_type column.

    Parameters
    ----------
    df: dataframe object

    Returns
    -------
    dataframe with fraud column fraud (1) or not fraud (0)
    '''

    df['fraud_target'] = df.apply(label_fraud, axis=1)
    #drop the acct type column since it is an output now.
    df = df.drop('acct_type', axis=1)
    return df

def label_fraud(row):
    '''Step through rows to test account type

    Parameters
    ----------
    row: row without fraud

    Returns
    -------
    row with fraud identified
    '''

    # if acct type has these labels it will get a 1 in the fraud column
    labels = ['fraudster', 'fraudster_event', 'fraudster_att']
    if row['acct_type'] in labels:
        return 1  #Fraud
    else:
        return 0  #No Fraud

def tfidf_vect(X, tok_func=None):
    tfidf = TfidfVectorizer(stop_words='english', max_features=500, tokenizer=tok_func)
    mtrx = tfidf.fit_transform(X)
    return mtrx

def tokenize(string):
    chars = [char for char in string if char not in punctuation and char in printable and char not in '0123456789']
    new_string = ''.join(chars)
    lowers = new_string.lower()

    tokens = word_tokenize(lowers)
    wordnet = WordNetLemmatizer()
    lems = [wordnet.lemmatize(t) for t in tokens]
    return lems

def stem(string):
    stemmer = PorterStemmer()
    chars = [char for char in string if char not in punctuation and char in printable and char not in '0123456789']
    new_string = ''.join(chars)
    words = new_string.lower().split()
    stems = [stemmer.stem(word) for word in words]
    return stems

def cv_tuning(X, y, clf_class, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    error = np.zeros(n_folds)
    f1_scores = np.zeros(n_folds)
    for i, (train_inds, val_inds) in enumerate(kf.split(X)):
        X_train = X[train_inds, :]
        y_train = y[train_inds]
        X_val = X[val_inds, :]
        y_val = y[val_inds]

        model = clf_class
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        f1 = f1_score(y_val, y_pred, average='weighted')
        f1_scores[i] = f1
        mse = mean_squared_error(y_val, y_pred)
        error[i] = mse

    mean_rmse = np.mean(np.sqrt(error))
    mean_f1 = np.mean(f1_scores)
    return mean_rmse, mean_f1

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
                                    scoring='f1_weighted')
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

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

from bs4 import BeautifulSoup

def get_processed_text(article_text):
    # soup = BeautifulSoup(article_text)
    # if soup.get_data() != None:
    #     article_text = soup.get_data()
    s = MLStripper()
    s.feed(article_text)
    article_text = s.get_data()

    tokenizer = RegexpTokenizer(r'\w+')
    raw = article_text.lower()
    tokens = tokenizer.tokenize(raw)

    # create English stop words list
    sw = set(stopwords.words('english'))
    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in sw]

    wordnet = WordNetLemmatizer()
    # stem token
    texts = " ".join([wordnet.lemmatize(i) for i in stopped_tokens])
    return texts

if __name__ == '__main__':
    # get data
    df = pd.read_csv('data/clean_data.csv')
    del df['Unnamed: 0']
    del df['acct_type']
    df = df.fillna('')
    y = df['fraud_target'].values

    columns_to_use = ['description', 'org_desc']
    for col in columns_to_use:
        X = df[col].values

        # df_small = clean_data('data/subset.json')
        # y = df_small.pop('fraud_target').values
        # X = df_small['description'].values

        indices = [i for i in df.index]
        X_train, X_test, y_train, y_test, X_indices, y_indices = train_test_split(X, y, indices, random_state=42, stratify=y)

        print('Creating train matrix...')
        mtrx_train = tfidf_vect(X_train, tokenize)
        print('Creating test matrix...')
        mtrx_test = tfidf_vect(X_test, tokenize)

        print('Creating model...')
        rf_clf = RandomForestClassifier()
        nb_clf = MultinomialNB()

        # mean_rmse, mean_f1 = cv_tuning(mtrx_train, y_train, rf_clf)

        rf_clf.fit(mtrx_train, y_train)
        y_pred = rf_clf.predict(mtrx_test)

        f1 = f1_score(y_test, y_pred, average='weighted')

        # gbc = GradientBoostingClassifier()
        #
        # gbc.fit(mtrx_train, y_train)
        # y_pred = gbc.predict(mtrx_test.todense())
        # f1_gbc = f1_score(y_test, y_pred, average='weighted')

        new_feature_train = rf_clf.predict_proba(mtrx_train.todense())[:,1]
        new_feature_test = rf_clf.predict_proba(mtrx_test.todense())[:,1]

        df[col].iloc[X_indices] = new_feature_train
        df[col].iloc[y_indices] = new_feature_test

        df[col] = pd.to_numeric(df[col])
        df.to_csv('data/nlp_data.csv')

        # rfc_grid = {"max_depth": [3, None],
        #           "max_features": [1, 3, 5, 8, 12],
        #           "min_samples_split": [2, 5, 8, 12],
        #           "min_samples_leaf": [2, 5, 8, 12],
        #           "bootstrap": [True, False],
        #           "criterion": ["gini", "entropy"]}
        #
        # rfc_best_params, rfc_best_model = gridsearch_with_output(RandomForestClassifier(), rfc_grid, mtrx_train, y_train)
        # print(get_score(y_test, rfc_best_model.predict(mtrx_test)))

        # gradient_boosting_grid = {'learning_rate': [0.1, 0.05, 0.02, 0.01],
        #                           'max_depth': [2, 4, 6],
        #                           'min_samples_leaf': [1, 2, 5, 10],
        #                           'max_features': [1.0, 0.3, 0.1],
        #                           'n_estimators': [400, 500, 600],
        #                           'random_state': [1]}
        # gbr_best_params, gbr_best_model = gridsearch_with_output(GradientBoostingClassifier(), gradient_boosting_grid, mtrx_train.todense(), y_train)

        # new_X = []
        # for description in X:
        #     new_X.append(get_processed_text(description))
        #
        # max_features = 1000
        # tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
        #                                 max_features=max_features,
        #                                 stop_words='english')
        # tf = tf_vectorizer.fit_transform(new_X)
        #
        # n_topics = 20
        # lda_model = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
        #                                       learning_method='online',
        #                                       learning_offset=50.,
        #                                       random_state=0)
        #
        # lda_model.fit(tf)
        # vis_data = pyLDAvis.sklearn.prepare(lda_model,tf, tf_vectorizer, R=n_topics, n_jobs=-1)
        # pyLDAvis.show(vis_data)
