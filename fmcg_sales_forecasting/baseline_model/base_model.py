import pandas as pd
import numpy as np
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
import dvc.api

class LastValueModel:
    def __init__(self, target_col):
        self.target_col = target_col

    def fit(self, df):
        self.value = df[self.target_col].iloc[-1]

    def predict(self, df):
        return np.full(len(df), self.value)


class MovingAverageModel:
    def __init__(self, target_col, window=3):
        self.target_col = target_col
        self.window = window

    def fit(self, df):
        self.hist = df[self.target_col].values

    def predict(self, df):
        preds = []
        for _ in range(len(df)):
            w = self.hist[-self.window:] if len(self.hist) >= self.window else self.hist
            preds.append(np.mean(w))
        return np.array(preds)


class SeasonalNaiveModel:
    def __init__(self, target_col, seasonality=12):
        self.target_col = target_col
        self.seasonality = seasonality

    def fit(self, df):
        self.hist = df[self.target_col].values

    def predict(self, df):
        preds = []
        for i in range(len(df)):
            idx = i - self.seasonality
            preds.append(self.hist[idx] if idx >= 0 else self.hist[-1])
        return np.array(preds)


class SARIMAModel:
    def __init__(self, target_col):
        self.target_col = target_col

    def fit(self, df):
        self.model = SARIMAX(
            df[self.target_col],
            order=(1,1,1),
            seasonal_order=(1,1,1,12),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)

    def predict(self, df):
        return self.model.forecast(len(df))


class XGBoostModel:
    def __init__(self, target_col):
        self.target_col = target_col
        self.model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05
        )

    def fit(self, df):
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        self.model.fit(X, y)

    def predict(self, df):
        return self.model.predict(df.drop(columns=[self.target_col]))

