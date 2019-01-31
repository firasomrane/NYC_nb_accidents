from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
import datetime, math
from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np

class FeatureExtractor(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X_df, y):
        return self
    
    def transform(self, X_df):
        X_augmented = self.extract_date_data(X_df)
        return X_augmented
    
    @staticmethod
    def extract_date_data(df):
        df_new = df.reset_index().copy()
        df_new['IsWeekend'] = df_new['DT'].apply(lambda x : 1 if (datetime.datetime.weekday(x) >= 5) else 0)
        df_new['Hour_cos'] = df_new['DT'].apply(lambda x : math.cos(x.hour * (2 * math.pi / 24)))
        df_new['Hour_sin'] = df_new['DT'].apply(lambda x : math.sin(x.hour * (2 * math.pi / 24)))
        df_new['Month_cos'] = df_new['DT'].apply(lambda x : math.cos(x.month * (2 * math.pi / 12)))
        df_new['Month_sin'] = df_new['DT'].apply(lambda x : math.sin(x.month * (2 * math.pi / 12)))
        df_new['WeekDay_cos'] = df_new['DT'].apply(lambda x : math.cos(datetime.datetime.weekday(x) * (2 * math.pi / 7)))
        df_new['WeekDay_sin'] = df_new['DT'].apply(lambda x : math.sin(datetime.datetime.weekday(x) * (2 * math.pi / 7)))
        df_new.set_index(['ZIPCODE','DT'], inplace=True )
        return df_new