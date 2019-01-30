import pandas as pd
import numpy as np
from google_drive_downloader import GoogleDriveDownloader as gdd
import rampwf as rw
from rampwf.score_types.base import BaseScoreType
from rampwf.score_types import relative_rmse
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score, precision_score
import os

problem_title = 'NYC_Num_of_accidents_prediction_'


Predictions = rw.prediction_types.make_regression(
    label_names=['n_collisions'])

workflow = rw.workflows.FeatureExtractorRegressor()


#--------------------------------------------
# Scoring
#--------------------------------------------

# Normalized root mean square error

class NRMSE(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = np.inf
    
    def __init__(self, name='nrmse',precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        nrmse = 0
        for val_true, val_pred in zip(y_true, y_pred):
            if(val_true != 0):
                if val_pred < val_true:
                    nrmse += (((val_pred - val_true)/val_true)* 1.5) ** 2
                else:
                    nrmse += ((val_pred - val_true)/val_true) ** 2
            else:
                nrmse += val_pred ** 2
        return np.sqrt(nrmse/y_true.shape[0])[0]


class Precision(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='prec', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        y_pred_binary = np.vectorize(lambda x : 0 if (x == 0) else 1)(y_pred)
        y_true_binary = np.vectorize(lambda x : 0 if (x == 0) else 1)(y_true)
        score = precision_score(y_true_binary, y_pred_binary)
        return score


class Recall(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='rec', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        y_pred_binary = np.vectorize(lambda x : 0 if (x == 0) else 1)(y_pred)
        y_true_binary = np.vectorize(lambda x : 0 if (x == 0) else 1)(y_true)
        score = recall_score(y_true_binary, y_pred_binary)
        return score


score_types = [
    
    # Normalized root mean square error
    NRMSE(name='nrmse', precision=2),
    # Precision and recall
    Precision(name='prec', precision=2),
    Recall(name='rec', precision=2)

]
#--------------------------------------------
# Cross validation
#--------------------------------------------


def get_cv(X, y):
    cv = KFold(n_splits=5, random_state=45)
    #print("get_cv = ", cv.split(X, y))
    return cv.split(X, y)

    
#--------------------------------------------
# Data reader
#--------------------------------------------

path = {'train_data' : '17PQOFwutGTz0-eYYxCXjgt-htcvR-9ne', 'train_target' : '1yVuDefFQ0SOTjreQwCzWSQMzR6xdbk8t', 
       'test_data' : '1r9HeG0mF8rFBq8-7BAd6hbNWfgaT76Zt', 'test_target': '13eDF-kbYVivMqQISvzbR4QWPRddgfMNB'}

def _read_data(name):
    gdd.download_file_from_google_drive(file_id=path[name+'_data'],
                                    dest_path='./data/' + name +'_data.csv')
    gdd.download_file_from_google_drive(file_id=path[name+'_target'],
                                    dest_path='./data/' + name +'_target.csv')

    data = pd.read_csv('./data/'+ name + '_data.csv', index_col=[0,1], parse_dates=['DT'])
    target = pd.read_csv('./data/'+name + '_target.csv', index_col=[0,1], parse_dates=['DT'])
    target = target['n_collisions'].values
    target = target.reshape((len(target), 1))
    return data, target

def get_train_data(path='.'):
    name = 'train'
    return _read_data(name)

def get_test_data(path='.'):
    name = 'test'
    return _read_data(name)


def _read_data_df(name):
    gdd.download_file_from_google_drive(file_id=path[name+'_data'],
                                    dest_path='./data/' + name +'_data.csv')
    gdd.download_file_from_google_drive(file_id=path[name+'_target'],
                                    dest_path='./data/' + name +'_target.csv')

    data = pd.read_csv('./data/'+ name + '_data.csv', index_col=[0,1], parse_dates=['DT'])
    target = pd.read_csv('./data/'+name + '_target.csv', index_col=[0,1], parse_dates=['DT'])
    return data, target

def get_train_data_df(path='.'):
    name = 'train'
    return _read_data_df(name)

def get_test_data_df(path='.'):
    name = 'test'
    return _read_data_df(name)

def get_zip_code():
    gdd.download_file_from_google_drive(file_id='1FkdgfZznyPJqZzcLnm2NYu1mwJDkkqmN',
                                        dest_path='./data/zipcode.geojson.zip', unzip = True)
