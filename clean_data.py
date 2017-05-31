import pandas as pd
import numpy as np

df_all = pd.read_json('data/data.json')

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
add_fraud_col(df_all).pop('fraud_target')
#del df_all['fraud_target']

# No good
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
del df_all['channels']
del df_all['user_age']

# Use later
del df_all['ticket_types']
del df_all['previous_payouts']
del df_all['description']
del df_all['email_domain']
del df_all['delivery_method']

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

df_all.to_csv('data/clean_data.csv')
