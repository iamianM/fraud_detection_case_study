from flask import Flask, render_template
import pickle
import requests
import pandas as pd
from bs4 import BeautifulSoup
import sys
sys.path.append('/home/ian/Galvanize/8_Special_Topics/fraud-detection-case-study')
from clean_data import clean_data
sys.path.append('/home/ian/Galvanize/8_Special_Topics/fraud-detection-case-study/web_app')
from fraud_model import FraudClassifier
from sklearn import metrics
from pymongo import MongoClient
import random

from flask import Flask, render_template, request
import webbrowser, threading, os

app = Flask(__name__)

# Create MongoClient
client = MongoClient()
# Initialize the Database
db = client['events']
tab = db['predicted_events']

def run_prediction():
    with open('../models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    url = 'http://galvanize-case-study-on-fraud.herokuapp.com/data_point'
    result = requests.get(url).json()
    df_all = pd.DataFrame.from_dict(result, orient='index').transpose()
    df = clean_data(df_all.copy(), training=False)

    del df['description']
    del df['org_desc']

    X = df.values
    print(X)
    prediction = int(model.predict(X)[0])
    prediction_proba = model.predict_proba(X)[0][1]

    insert = df_all.to_dict(orient='records')[0]
    insert['prediction'] = prediction
    insert['prediction_proba'] = prediction_proba

    if not bool(tab.find({'object_id': df_all['object_id'][0]}).count()):
        tab.insert_one(insert)
    return prediction, prediction_proba, df_all

def run_previous_prediction(temp):
    print(int(temp))
    results = tab.find({'prediction': int(temp)})
    r = random.randint(0, results.count()-1)
    result = results[r]
    df_all = pd.DataFrame.from_dict(result, orient='index').transpose()
    # df_all['object_id'] = 0
    df = clean_data(df_all.copy(), training=False)
    prediction = df['prediction'][0]
    prediction_proba = df['prediction_proba'][0]

    return prediction, prediction_proba, df_all

# home page
@app.route('/')
def index():
    return render_template('home.html')

# Button for prediction page
@app.route('/input')
def results():
    return render_template('input.html')

# predict page
@app.route("/predict", methods=['POST'])
def predict():
    fraud, prediction_proba, df = run_prediction()
    if fraud:
        out = 'FRAUD DETECTED!!!     Fraud has been predicted with a probability of '+str(prediction_proba)
    else:
        out = 'No Fraud Detected.    No Fraud has been predicted with a probability of '+str(prediction_proba)
    description = df['description'][0]
    org_name = df['org_name'][0]
    org_desc = df['org_desc'][0]
    return render_template('prediction.html', out=out, description=description, org_name=org_name, org_desc=org_desc)

@app.route("/fraud", methods=['POST'])
def fraud():
    fraud, prediction_proba, df = run_previous_prediction(1)
    out = 'FRAUD DETECTED!!!     Fraud has been predicted with a probability of '+str(prediction_proba)
    description = df['description'][0]
    org_name = df['org_name'][0]
    org_desc = df['org_desc'][0]
    return render_template('prediction.html', out=out, description=description, org_name=org_name, org_desc=org_desc)

@app.route("/nonfraud", methods=['POST'])
def nonfraud():
    fraud, prediction_proba, df = run_previous_prediction(0)
    out = 'No Fraud Detected.    No Fraud has been predicted with a probability of '+str(prediction_proba)
    description = df['description'][0]
    org_name = df['org_name'][0]
    org_desc = df['org_desc'][0]
    return render_template('prediction.html', out=out, description=description, org_name=org_name, org_desc=org_desc)

# graphs page
@app.route('/graphs')
def graphs():
    return render_template('graphs.html')

# topics page
@app.route('/desctopics')
def desctopics():
    return render_template('pyldadescription.html')

# topics page
@app.route('/orgtopics')
def orgtopics():
    return render_template('pyldaorg_desc.html')


# about page
@app.route('/about')
def about():
    return render_template('about.html')

# contact page
@app.route('/contact')
def contact():
    return render_template('contact.html')

# @app.route('/more/')
# def more():
#     return render_template('starter_template.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8105, debug=True)
