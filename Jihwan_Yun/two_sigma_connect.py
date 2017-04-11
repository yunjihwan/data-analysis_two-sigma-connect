# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 22:17:02 2017

@author: rnfmf
"""

import os
import sys
import string
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn import preprocessing
from sklearn.feature_extraction.text import  CountVectorizer
from scipy.stats import boxcox
from scipy import sparse

from xgboost import plot_importance
from matplotlib import pyplot

data_path = "data/"
train_file = data_path + "train.json"
test_file = data_path + "test.json"
train = pd.read_json(train_file)
test = pd.read_json(test_file)

listing_id = test.listing_id.values

y_map = {'low': 2, 'medium': 1, 'high': 0}
train['interest_level'] = train['interest_level'].apply(lambda x: y_map[x])
y_train = train.interest_level.values

train = train.drop(['listing_id', 'interest_level'], axis=1)
test = test.drop('listing_id', axis=1)

ntrain = train.shape[0]

train_test = pd.concat((train, test), axis=0).reset_index(drop=True)
train_test.head()

features_to_use  = ["bathrooms","bedrooms","building_id", "created","latitude", "description",
                    "listing_id","longitude","manager_id", "price", "features", "display_address", 
                    "street_address","feature_count","photo_count", "interest_level"]

train_test["price_per_bed"] = train_test["price"] / train_test["bedrooms"]
train_test["room_diff"] = train_test["bedrooms"] - train_test["bathrooms"] 
train_test["room_sum"] = train_test["bedrooms"] + train_test["bathrooms"] 
train_test["room_price"] = train_test["price"] / train_test["room_sum"]
train_test["bed_ratio"] = train_test["bedrooms"] / train_test["room_sum"]
train_test["bed_bath_sum"] = train_test["bedrooms"] / train_test["bathrooms"] 
train_test = train_test.fillna(-1).replace(np.inf, -1)

train_test["photo_count"] = train_test["photos"].apply(len)
train_test["feature_count"] = train_test["features"].apply(len)

#log transform
train_test["photo_count"] = np.log(train_test["photo_count"] + 1)
train_test["feature_count"] = np.log(train_test["feature_count"] + 1)
train_test["price"] = np.log(train_test["price"] + 1)
train_test["price_per_bed"] = np.log(train_test["price_per_bed"] + 1)
#train_test["room_diff"] = np.log(train_test["room_diff"] + 1) #-inf값 생성됨
train_test["room_sum"] = np.log(train_test["room_sum"] + 1)
train_test["room_price"] = np.log(train_test["room_price"] + 1)
train_test["bed_ratio"] = np.log(train_test["bed_ratio"] + 1)
train_test["bed_bath_sum"] = np.log(train_test["bed_bath_sum"] + 1)

#date time
train_test["created"] = pd.to_datetime(train_test["created"])
train_test["passed"] = train_test["created"].max() - train_test["created"]

train_test["created_year"] = train_test["created"].dt.year
train_test["created_month"] = train_test["created"].dt.month
train_test["created_day"] = train_test["created"].dt.day
train_test["created_wday"] = train_test["created"].dt.dayofweek
train_test["created_yday"] = train_test["created"].dt.dayofyear
train_test["created_hour"] = train_test["created"].dt.hour

#for text pre-processing
fmt = lambda s: s.replace("\u00a0", "").strip().lower()
 
#description
train_test['desc'] = train_test['description'].apply(fmt)
train_test['desc'] = train_test['desc'].apply(lambda x: x.replace('<p><a  website_redacted ', ''))
train_test['desc'] = train_test['desc'].apply(lambda x: x.replace('!<br /><br />', ''))

string.punctuation.__add__('!!')
string.punctuation.__add__('(')
string.punctuation.__add__(')')
remove_punct_map = dict.fromkeys(map(ord, string.punctuation))

train_test['desc'] = train_test['desc'].apply(lambda x: x.translate(remove_punct_map))
train_test['desc_letters_count'] = train_test['desc'].apply(lambda x: len(x.strip()))
train_test['desc_words_count'] = train_test['desc'].apply(lambda x: 0 if len(x.strip()) == 0 else len(x.split(' ')))

 #address
address_map = {
    'w': 'west',
    'st.': 'street',
    'ave': 'avenue',
    'st': 'street',
    'e': 'east',
    'n': 'north',
    's': 'south'
}

def address_map_func(s):
    s = s.split(' ')
    out = []
    for x in s:
        if x in address_map:
            out.append(address_map[x])
        else:
            out.append(x)
    return ' '.join(out)

train_test['dis_address'] = train_test['display_address'].apply(fmt)
train_test['dis_address'] = train_test['dis_address'].apply(lambda x: x.translate(remove_punct_map))
train_test['dis_address'] = train_test['dis_address'].apply(lambda x: address_map_func(x))

new_cols = ['street', 'avenue', 'east', 'west', 'north', 'south']

for col in new_cols:
    train_test[col] = train_test['dis_address'].apply(lambda x: 1 if col in x else 0)
    
train_test['other_address'] = train_test[new_cols].apply(lambda x: 1 if x.sum() == 0 else 0, axis=1)    

#Extract top feature
def create_top_features(df, top_num, feat_name, count_dict):
    percent = 100
    for i in range(1, top_num):
        df['top_' + str(i) + '_' + feat_name] = df[feat_name].apply(lambda x: 1 if x in count_dict.index.values[count_dict.values >= np.percentile(count_dict.values, percent - i)] else 0)
    return df    

#Manage id
managers_count = train_test['manager_id'].value_counts()
train_test = create_top_features(train_test, 10, "manager_id", managers_count)
#Building id
buildings_count = train_test['building_id'].value_counts()
train_test = create_top_features(train_test, 10, "building_id", buildings_count)
#zero building id
train_test['Zero_building_id'] = train_test['building_id'].apply(lambda x: 1 if x == '0' else 0)

def designate_single_observations(df, column):
    ps = df[column]
    grouped = ps.groupby(ps).size().to_frame().rename(columns={0: "size"})
    df.loc[df.join(grouped, on=column, how="left")["size"] <= 1, column] = -1
    return df

# Special designation for building_ids, manager_ids, display_address with only 1 observation
for col in ('building_id', 'manager_id', 'display_address'):
    train_test = designate_single_observations(train_test, col)
    
#features
train_test['feats'] = train_test['features']
train_test['features_count'] = train_test['feats'].apply(lambda x: len(x))
train_test['feats'] = train_test['feats'].apply(lambda x: ' '.join(x).lower())

train_test['feats'] = train_test['feats'].apply(lambda x: x.replace("dogs", "dog"))
train_test['feats'] = train_test['feats'].apply(lambda x: x.replace("cats", "cat"))
train_test['feats'] = train_test['feats'].apply(lambda x: x.replace("no fee", "nofee"))
train_test['feats'] = train_test['feats'].apply(lambda x: x.replace("no-fee", "nofee"))
train_test['feats'] = train_test['feats'].apply(lambda x: x.replace("no_fee", "nofee"))
train_test['feats'] = train_test['feats'].apply(lambda x: x.replace("reduced_fee", "lowfee"))
train_test['feats'] = train_test['feats'].apply(lambda x: x.replace("reduced fee", "lowfee"))
train_test['feats'] = train_test['feats'].apply(lambda x: x.replace("hardwood", "parquet"))
train_test['feats'] = train_test['feats'].apply(lambda x: x.replace("doorman", "concierge"))
train_test['feats'] = train_test['feats'].apply(lambda x: x.replace("housekeep", "concierge"))
train_test['feats'] = train_test['feats'].apply(lambda x: x.replace("in_super", "concierge"))
train_test['feats'] = train_test['feats'].apply(lambda x: x.replace("pre_war", "prewar"))
train_test['feats'] = train_test['feats'].apply(lambda x: x.replace("pre war", "prewar"))
train_test['feats'] = train_test['feats'].apply(lambda x: x.replace("pre-war", "prewar"))
train_test['feats'] = train_test['feats'].apply(lambda x: x.replace("lndry", "laundry"))
train_test['feats'] = train_test['feats'].apply(lambda x: x.replace("gym", "health"))
train_test['feats'] = train_test['feats'].apply(lambda x: x.replace("fitness", "health"))
train_test['feats'] = train_test['feats'].apply(lambda x: x.replace("training", "health"))
train_test['feats'] = train_test['feats'].apply(lambda x: x.replace("swimming", "health"))
train_test['feats'] = train_test['feats'].apply(lambda x: x.replace("train", "transport"))
train_test['feats'] = train_test['feats'].apply(lambda x: x.replace("subway", "transport"))
train_test['feats'] = train_test['feats'].apply(lambda x: x.replace("heat water", "utilities"))
train_test['feats'] = train_test['feats'].apply(lambda x: x.replace("water included", "utilities"))

c_vect = CountVectorizer(stop_words='english', max_features=200, ngram_range=(1, 1))
c_vect.fit(train_test['feats'])

c_vect_sparse_1 = c_vect.transform(train_test['feats'])
c_vect_sparse1_cols = c_vect.get_feature_names()



#remove useless features
train_test.drop(['created'], axis=1, inplace=True)
train_test.drop(['description'], axis=1, inplace=True)
train_test.drop(['desc'], axis=1, inplace=True)
train_test.drop(['street_address'], axis=1, inplace=True)
train_test.drop(['display_address'], axis=1, inplace=True)
train_test.drop(['features'], axis=1, inplace=True)
train_test.drop(['feats'], axis=1, inplace=True)
train_test.drop(['photos'], axis=1, inplace=True)
train_test.drop(['manager_id'], axis=1, inplace=True)
train_test.drop(['building_id'], axis=1, inplace=True)
train_test.drop(['dis_address'], axis=1, inplace=True) #확인 필요
train_test.drop(['passed'], axis=1, inplace=True) #확인 필요

#현재 feature 데이터 확인
for col in train_test.columns.values:
    print(train_test[col].head(2))
    
#train model
features = list(train_test.columns)
train_test_cv1_sparse = sparse.hstack((train_test.astype(float), c_vect_sparse_1)).tocsr()

x_train = train_test_cv1_sparse[:ntrain, :]
x_test = train_test_cv1_sparse[ntrain:, :]
features += c_vect_sparse1_cols

SEED = 777
NFOLDS = 5

params = {
    'eta':.05,
    'max_depth': 5,
    'colsample_bytree':.89,
    'subsample':.9,
    'seed':0,
    'nthread':16,
    'objective':'multi:softprob',
    'eval_metric':'mlogloss',
    'num_class':3,
    'silent':1
}

#create model
dtrain = xgb.DMatrix(data=x_train, label=y_train)
dtest = xgb.DMatrix(data=x_test)

bst = xgb.cv(params, dtrain, 50000, NFOLDS, early_stopping_rounds=50, verbose_eval=25)
best_rounds = np.argmin(bst['test-mlogloss-mean'])
bst = xgb.train(params, dtrain, best_rounds)

#predict
preds = pd.DataFrame(bst.predict(dtest))
cols = ['high', 'medium', 'low']
preds.columns = cols
preds['listing_id'] = listing_id
preds.to_csv('my_preds3.csv', index=None)

#Show feature importance
bst.get_fscore()
mapper = {'f{0}'.format(i): v for i, v in enumerate(features)}
mapped = {mapper[k]: v for k, v in bst.get_fscore().items()}
xgb.plot_importance(mapped, max_num_features=20, color='red',  height=0.6)
pyplot.figure(figsize=(18,18))
pyplot.show()