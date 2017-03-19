"""
Random forest
Ref: https://www.kaggle.com/aikinogard/two-sigma-connect-rental-listing-inquiries/random-forest-starter-with-numerical-features

"""

import datetime

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

train_file = 'dataset/train.json'
test_file = 'dataset/test.json'


def add_features(df):
    # add naive features
    df['num_photos'] = df['photos'].apply(len)
    df['num_features'] = df['features'].apply(len)
    df['num_description_words'] = df['description'].apply(lambda x: len(x.split(' ')))

    df['created'] = pd.to_datetime(df['created'])
    df['created_year'] = df['created'].dt.year
    df['created_month'] = df['created'].dt.month
    df['created_day'] = df['created'].dt.day

    return df

# Build a random forest model
train = pd.read_json(train_file)
train = add_features(train)
numeric_features = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price', 'num_photos',
                    'num_features', 'num_description_words', 'created_year', 'created_month', 'created_day']

x = train[numeric_features]
y = train['interest_level']

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.33)

clf = RandomForestClassifier(n_estimators=1000)
clf.fit(x_train, y_train)
y_val_pred = clf.predict_proba(x_val)
loss = log_loss(y_true=y_val, y_pred=y_val_pred)
print('loss={}'.format(loss))

# Make prediction
test = pd.read_json(test_file)
test = add_features(test)
x = test[numeric_features]
y = clf.predict_proba(x)

label2index = {label: i for i, label in enumerate(clf.classes_)}

submission = pd.DataFrame()
submission['listing_id'] = test['listing_id']
for label in ['high', 'medium', 'low']:
    submission[label] = y[:, label2index[label]]
submission_file = 'submission/{}_randomforest.csv'\
    .format(datetime.datetime.now().strftime('%Y%m%d'))
submission.to_csv(submission_file, index=False)