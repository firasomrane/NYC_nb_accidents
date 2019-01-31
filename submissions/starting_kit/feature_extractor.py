from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import datetime
import math
import pandas as pd

class FeatureExtractor(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X_df, y):        
        return self

    def transform(self, X_df):
        X_new = X_df.copy()
        X_new = extract_date_and_hour(X_new)
        return X_new


def extract_date_and_hour(df):
    df_new = df.reset_index().copy()
    df_new['Hour'] = df_new['DT'].apply(lambda x : x.hour )
    dfDummies = pd.get_dummies(df_new['Hour'], prefix = 'Hour')
    df_ = pd.concat([df.reset_index(), dfDummies], axis=1, sort=False)
    df_['WeekDay'] = df_new['DT'].apply(lambda x : 1 if (datetime.datetime.weekday(x) < 5) else 0 )
    df_.set_index(['ZIPCODE','DT'], inplace =True )
    return df_

def extract_date_data(df):
    df_new = df.reset_index().copy()
    df_new['Hour'] = df_new['DT'].apply(lambda x : x.hour )
    df_new['Month'] = df_new['DT'].apply(lambda x : x.month )
    df_new['WeekDay'] = df_new['DT'].apply(lambda x : datetime.datetime.weekday(x))
    df_new['IsWeekend'] = df_new['WeekDay'].apply(lambda x : 1 if (x >= 5) else 0)
    df_new['Hour_cos'] = df_new['Hour'].apply(lambda x : math.cos(x * (2 * math.pi / 24)))
    df_new['Hour_sin'] = df_new['Hour'].apply(lambda x : math.sin(x * (2 * math.pi / 24)))
    df_new['Month_cos'] = df_new['Month'].apply(lambda x : math.cos(x * (2 * math.pi / 12)))
    df_new['Month_sin'] = df_new['Month'].apply(lambda x : math.sin(x * (2 * math.pi / 12)))
    df_new['WeekDay_cos'] = df_new['WeekDay'].apply(lambda x : math.cos(x * (2 * math.pi / 7)))
    df_new['WeekDay_sin'] = df_new['WeekDay'].apply(lambda x : math.sin(x * (2 * math.pi / 7)))
    df_new.drop(['Hour', 'Month', 'WeekDay'], axis=1, inplace=True)
    df_new.set_index(['ZIPCODE','DT'], inplace=True )
    return df_new

# Fonction pour réduire le train
def downsample_data(df, y, factor):
    df_new = df.copy()
    df_new['n_collisions'] = y
    df_new  = df_new.drop(df_new.query('n_collisions == 0').sample(frac=factor).index)
    df =  df_new.iloc[:,:-1]
    y = (df_new.iloc[:,-1]).values
    return df, y.reshape((len(y), 1))
