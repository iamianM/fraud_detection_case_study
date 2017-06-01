import pandas as pd
import numpy as np
from html.parser import HTMLParser
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer



def add_fraud_col(df):
    '''
    Add fraud_target Column based on acct_type column.
    Input: data frame
    Returns: data frame with fraud column fraud (1) or not fraud (0)
    '''

    df['fraud_target'] = df.apply(label_fraud, axis=1)
    #drop the acct type column since it is an output now.
    df_withfraud = df.drop('acct_type', axis=1)
    return df_withfraud

def label_fraud(row):
    '''
    Step through rows to test account type
    Input: row without fraud
    Returns: row with fraud identified. If acct_type has  'fraudster', 'fraudster_event' or  'fraudster_att'
    '''

    # if acct type has these labels it will get a 1 in the fraud column
    labels = [ 'fraudster', 'fraudster_event', 'fraudster_att']

    if row['acct_type'] in labels:
        return 1  #Fraud
    else:
        return 0  #No Fraud

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

def clean_data(df_all, training=True, save=False):

    print('Deleting columns...')
    # No good
    del df_all['object_id']
    del df_all['has_header']
    del df_all['org_name']
    # del df_all['org_desc']
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
    del df_all['channels']
    del df_all['user_age']

    # Use later
    del df_all['ticket_types']
    del df_all['previous_payouts']
    # del df_all['description']
    del df_all['email_domain']
    del df_all['delivery_method']

    print('Processing text data...')
    processed_text = []
    for i, text in enumerate(df_all['description']):
        processed_text.append(get_processed_text(text))
    df_all['description'] = processed_text
    processed_text = []
    for i, text in enumerate(df_all['org_desc']):
        processed_text.append(get_processed_text(text))
    df_all['org_desc'] = processed_text

    print('Engineering date data...')
    df_all['time_to_payout'] = df_all.apply(lambda x: x['approx_payout_date'] - x['event_end'], axis=1)
    df_all['time_till_event'] = df_all.apply(lambda x: x['event_created'] - x['user_created'], axis=1)
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

    print('Doing stuff...')
    df_all['is_country_same'] = df_all.apply(lambda x: 1 if x['country'] == x['venue_country'] else 0, axis=1)
    del df_all['country']
    del df_all['venue_country']

    df_all['listed'] = df_all.apply(lambda x: 1 if x['listed'] == 'y' else 0, axis=1)

    print('Dummying payout types...')
    if len(df_all['payout_type'].unique()) > 2:
        dummies = pd.get_dummies(df_all['payout_type'], drop_first=True)
        df_all['CHECK'] = dummies['CHECK']
        df_all['ACH'] = dummies['ACH']
    else:
        if df_all['payout_type'][0] == 'CHECK':
            df_all['CHECK'] = 1
            df_all['ACH'] = 0
        elif df_all['payout_type'][0] == 'ACH':
            df_all['ACH'] = 1
            df_all['CHECK'] = 0
        else:
            df_all['ACH'] = 0
            df_all['CHECK'] = 0

    del df_all['payout_type']

    if training:
        df_all = add_fraud_col(df_all)

    if save:
        print('Saving clean data...')
        df_all.to_csv('data/clean_data.csv')

    return df_all

if __name__ == '__main__':
    df = pd.read_json('data/data.json')
    clean_data(df)
