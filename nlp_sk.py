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
from sklearn.metrics import mean_squared_error, accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
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


def tfidf_vect(X):
    tfidf = TfidfVectorizer(stop_words='english', max_features=500)
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
    # del df['acct_type']
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

        # print('Creating train matrix...')
        # mtrx_train = tfidf_vect(X_train)
        # print('Creating test matrix...')
        # mtrx_test = tfidf_vect(X_test)
        #
        # from sklearn.pipeline import Pipeline
        # from sklearn.feature_extraction.text import CountVectorizer
        # count_vect = CountVectorizer()
        # X_train_counts = count_vect.fit_transform(X_train)
        # from sklearn.feature_extraction.text import TfidfTransformer
        # tfidf_transformer = TfidfTransformer()
        # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        # from sklearn.naive_bayes import MultinomialNB
        # clf = MultinomialNB().fit(X_train_tfidf, y_train)
        # X_new_counts = count_vect.transform(X_test)
        # X_new_tfidf = tfidf_transformer.transform(X_new_counts)
        #
        # predicted = clf.predict(X_new_tfidf)
        # f1 = f1_score(y_test, predicted)
        # print('clf f1 score: '+str(f1))
        # fpr, tpr, thresholds = roc_curve(y_test, predicted)
        # area = auc(fpr, tpr)
        # print('clf AUC: '+ str(area))
        # from sklearn import metrics
        # print(metrics.classification_report(y_test, predicted))
        #
        #
        # from sklearn.linear_model import SGDClassifier
        # text_clf = Pipeline([('vect', CountVectorizer()),
        #     ('tfidf', TfidfTransformer()),
        #     ('clf', SGDClassifier(loss='hinge', penalty='l2',
        #     alpha=1e-3, n_iter=5, random_state=42)),
        # ])
        # _ = text_clf.fit(X_train, y_train)
        # predicted = text_clf.predict(X_test)
        # print(metrics.classification_report(y_test, predicted))
        #
        # from sklearn.model_selection import GridSearchCV
        # parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
        #     'tfidf__use_idf': (True, False),
        #     'clf__alpha': (1e-2, 1e-3),
        # }
        # gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
        # gs_clf = gs_clf.fit(X_train, y_train)
        # predicted = gs_clf.predict(X_test)
        # print(metrics.classification_report(y_test, predicted))
        #
        # f1 = f1_score(y_test, predicted)
        # print('sdg f1 score: '+str(f1))
        # fpr, tpr, thresholds = roc_curve(y_test, predicted)
        # area = auc(fpr, tpr)
        # print('sdg AUC: '+ str(area))
        #
        # print('Creating model...')
        # rf_clf = RandomForestClassifier()
        # nb_clf = MultinomialNB()
        #
        # # mean_rmse, mean_f1 = cv_tuning(mtrx_train, y_train, rf_clf)
        #
        # rf_clf.fit(mtrx_train, y_train)
        # y_pred = rf_clf.predict(mtrx_test)
        #
        # f1 = f1_score(y_test, y_pred)
        # print('rf f1 score: '+str(f1))
        # fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        # area = auc(fpr, tpr)
        # print('rf AUC: '+ str(area))
        #
        # nb_clf.fit(mtrx_train, y_train)
        # y_pred = nb_clf.predict(mtrx_test)
        #
        # f1 = f1_score(y_test, y_pred)
        # print('nb f1 score: '+str(f1))
        # fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        # area = auc(fpr, tpr)
        # print('nb AUC: '+ str(area))
        #
        # gbc = GradientBoostingClassifier()
        #
        # gbc.fit(mtrx_train.todense(), y_train)
        # y_pred = gbc.predict(mtrx_test.todense())
        #
        # f1 = f1_score(y_test, y_pred)
        # print('gbc f1 score: '+str(f1))
        # fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        # area = auc(fpr, tpr)
        # print('gbc AUC: '+ str(area))
        #
        # svc = SVC()
        #
        # svc.fit(mtrx_train, y_train)
        # y_pred = svc.predict(mtrx_test.todense())
        #
        # f1 = f1_score(y_test, y_pred)
        # print('svc f1 score: '+str(f1))
        # fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        # area = auc(fpr, tpr)
        # print('svc AUC: '+ str(area))
        #
        # new_feature_train = gbc.predict_proba(mtrx_train.todense())[:,1]
        # gbc.fit(mtrx_test, y_test)
        # new_feature_test = gbc.predict_proba(mtrx_test.todense())[:,1]
        #
        # df[col].iloc[X_indices] = new_feature_train
        # df[col].iloc[y_indices] = new_feature_test
        #
        # df[col] = pd.to_numeric(df[col])
        # df.to_csv('data/nlp_data.csv')

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

        new_X = []
        for description in X:
            new_X.append(get_processed_text(description))

        max_features = 1000
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                        max_features=max_features,
                                        stop_words='english')
        tf = tf_vectorizer.fit_transform()

        n_topics = 20
        lda_model = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                              learning_method='online',
                                              learning_offset=50.,
                                              random_state=0)

        lda_model.fit(tf)
        vis_data = pyLDAvis.sklearn.prepare(lda_model,tf, tf_vectorizer, R=n_topics, n_jobs=-1)
        # pyLDAvis.show(vis_data)
        pyLDAvis.save_html(vis_data, 'web_app/templates/pylda'+str(col)+'.html')
