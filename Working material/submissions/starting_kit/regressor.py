import xgboost as xgb
from sklearn.base import BaseEstimator

class Regressor(BaseEstimator):
# To optimize performance we chose the XGBoost model and selected parmeters with GridSearch

    def __init__(self):
        self.reg = xgb.XGBRegressor(colsample_bytree = 0.7, learning_rate = 0.05, 
                                    max_depth = 7, min_child_weight = 4, n_estimators = 5000,
                                    nthread = 4, objective = 'reg:linear', silent = 1, 
                                    subsample = 0.7)
    def fit(self, X, y):
        self.reg.fit(X, y)      

    def predict(self, X):
        return self.reg.predict(X)