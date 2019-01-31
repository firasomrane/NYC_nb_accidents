from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn import metrics
import datetime, math
from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np

class Regressor(BaseEstimator):
    def __init__(self):
        self.rfr = make_pipeline(StandardScaler(), RandomForestRegressor())  #Scaling in pipeline

    def fit(self, X, y):
        X_new, y_new = self.downsample_data(X, y, 0.7) #Change the downsample factor if needed
        nrmse_scorer = metrics.make_scorer(self.nrmse, greater_is_better=False) #Custom scorer
        param_grid = {
           "standardscaler": [StandardScaler(), RobustScaler(), QuantileTransformer()], #Try different scalers
           "randomforestregressor__n_estimators" : [100, 300, 500],
           "randomforestregressor__max_depth" : [5, 10, 20],
           "randomforestregressor__min_samples_leaf" : [4, 6, 8],
           "randomforestregressor__max_features": ['auto', 'sqrt', 'log2']}
        CV_rfr = RandomizedSearchCV(estimator=self.rfr, param_distributions=param_grid, cv=4, scoring=nrmse_scorer, n_jobs=1)
        CV_rfr.fit(X_new, y_new) #Perform the hyperparameters tuning
        self.best_rfr = CV_rfr.best_estimator_ #Select the best regressor
        best_score = CV_rfr.best_score_
        print("Best train score found: " + str(best_score))
        print("Best parameters found: " + str(CV_rfr.best_params_))

    def predict(self, X):
        y_pred = self.best_rfr.predict(X).round().astype(int) #Predict with the best regressor found during the RandomizedSearch
        return y_pred.reshape((len(y_pred), 1))
    
    @staticmethod
    def downsample_data(df, y, factor):
        df_new = df.copy()
        df_new['n_collisions'] = y
        df_new  = df_new.drop(df_new.query('n_collisions == 0').sample(frac=factor).index)
        df =  df_new.iloc[:,:-1]
        y = (df_new.iloc[:,-1]).values
        return df, y.reshape((len(y), 1))
    
    @staticmethod
    def nrmse(y_true, y_pred):
        nrmse = 0
        y_pred_round = y_pred.round()
        for val_true, val_pred in zip(y_true, y_pred_round):
            if(val_true != 0):
                if val_pred < val_true:
                    nrmse += (((val_pred - val_true)/val_true)* 1.5) ** 2
                else:
                    nrmse += ((val_pred - val_true)/val_true) ** 2
            else:
                nrmse += val_pred ** 2
        return np.sqrt(nrmse/y_true.shape[0])[0]
