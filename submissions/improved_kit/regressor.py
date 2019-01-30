from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import datetime, math
from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np



class Regressor(BaseEstimator):
    def __init__(self):
        self.rfr = RandomForestRegressor()

    def fit(self, X, y):
        self.scaler = make_column_transformer(
            (StandardScaler(), list(X)) #Scale the features with a standard scaler
        )
        X_scaled = self.scaler.fit_transform(X) #fit the scaler and transform the data
        X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns) #Restore shape of dataframe after StandardScaler
        X_new, y_new = self.downsample_data(X_scaled_df, y, 0.7) #Change the downsample factor if needed
        nrmse_scorer = metrics.make_scorer(self.nrmse, greater_is_better=False) #Custom scorer
        param_grid = { 
           "n_estimators" : [100, 300, 500],
           "max_depth" : [5, 10, 20],
           "min_samples_leaf" : [4, 6, 8],
           "max_features": ['auto', 'sqrt', 'log2']}
        CV_rfr = RandomizedSearchCV(estimator=self.rfr, param_distributions=param_grid, cv=4, scoring=nrmse_scorer, n_jobs=-1)
        CV_rfr.fit(X_new, y_new.reshape((len(y), 1)))
        #Perform the hyperparameters tuning
        self.best_rfr = CV_rfr.best_estimator_ #Select the best regressor
        best_score = CV_rfr.best_score_
        print("Best train score found: " + str(best_score))

    def predict(self, X):
        X_scaled = self.scaler.transform(X) #transform the data based on the previous fit of the scaler
        y_pred = self.best_rfr.predict(X_scaled).round().astype(int)
        return y_pred.reshape((len(y_pred), 1)) #Predict with the best regressor found during the RandomizedSearch
    
    @staticmethod
    def downsample_data(df, y, factor):
        df_new = pd.concat([df, y], axis=1, sort=False)
        df_new  = df_new.drop(df_new.query('n_collisions == 0').sample(frac=factor).index)
        df =  df_new.iloc[:,:-1]
        y = df_new.iloc[:,-1]
        return df, y
    
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
        return np.sqrt(nrmse/y_true.shape[0])