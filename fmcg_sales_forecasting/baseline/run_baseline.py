import pandas as pd
import numpy as np
import hydra
import mlflow
import os
import pickle
from ..baseline_model.base_model import LastValueModel, MovingAverageModel, SARIMAModel, SeasonalNaiveModel, XGBoostModel
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import dvc.api

import mlflow

def setup_mlflow(cfg):
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def smape(y_true, y_pred):
    return 100 * np.mean(
        2 * np.abs(y_pred - y_true) /
        (np.abs(y_pred) + np.abs(y_true) + 1e-8)
    )

def wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


def compute_metrics(y_true, y_pred):
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "WAPE": wape(y_true, y_pred),
    }



def split_by_year(df, cfg):
    date_col = cfg.preprocessing.date_features.date_col
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if cfg.preprocessing.date_features.add_year:
        df["Year"] = df[date_col].dt.year
    return (
        df[df["Year"].between(2018, 2022)],
        df[df["Year"] == 2023],
        df[df["Year"] == 2024],
    )



def preprocess(df, cfg, scaler=None, ohe=None, fit=True):
    df = df.copy()

    target = cfg.preprocessing.target_col
    row_col = cfg.preprocessing.row_label_col

    # missing drop
    thresh = cfg.preprocessing.missing_values.drop_threshold_ratio * len(df)
    df = df.loc[:, df.isna().sum() <= thresh]

    # fill missing
    if cfg.preprocessing.missing_values.fill_strategy == "group_mean":
        for col in df.columns:
            if df[col].isna().any() and col != row_col:
                df[col] = df.groupby(row_col)[col].transform(lambda x: x.fillna(x.mean()))

    date_col = cfg.preprocessing.date_features.date_col
    if cfg.preprocessing.date_features.enabled:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        if cfg.preprocessing.date_features.add_year:
            df["Year"] = df[date_col].dt.year

        if cfg.preprocessing.date_features.add_month_sin_cos:
            m = df[date_col].dt.month
            df["Month_sin"] = np.sin(2 * np.pi * m / 12)
            df["Month_cos"] = np.cos(2 * np.pi * m / 12)

    if cfg.preprocessing.target_transform.log_transform:
        df[target] = np.log1p(df[target])

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in [target] + cfg.preprocessing.scaling.exclude_cols]

    if cfg.preprocessing.scaling.enabled:
        if fit:
            scaler = StandardScaler()
            df[num_cols] = scaler.fit_transform(df[num_cols])
        else:
            df[num_cols] = scaler.transform(df[num_cols])

    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in cfg.preprocessing.categorical.drop_cols]

    if cfg.preprocessing.categorical.enabled and cat_cols:
        if fit:
            ohe = OneHotEncoder(
                sparse_output=cfg.preprocessing.categorical.ohe.sparse,
                drop=cfg.preprocessing.categorical.ohe.drop,
                handle_unknown=cfg.preprocessing.categorical.ohe.handle_unknown,
            )
            enc = ohe.fit_transform(df[cat_cols])
        else:
            enc = ohe.transform(df[cat_cols])

        enc_df = pd.DataFrame(
            enc,
            columns=ohe.get_feature_names_out(cat_cols),
            index=df.index
        )
        df = pd.concat([df.drop(columns=cat_cols), enc_df], axis=1)

    return df, scaler, ohe


def download_data(path_in_dvc: str) -> pd.DataFrame:
    with dvc.api.open(path_in_dvc, repo=".") as f:
        df = pd.read_csv(f)
    return df

#  MLflow 

class MLflowLogger:
    def __init__(self, experiment_name):
        mlflow.set_experiment(experiment_name)

    def start_run(self, name):
        return mlflow.start_run(run_name=name)

    def log_params(self, params):
        mlflow.log_params(params)

    def log_metrics(self, metrics, prefix=""):
        for k, v in metrics.items():
            mlflow.log_metric(prefix + k, float(v))

    def log_model_checkpoint(self, model, name):
        os.makedirs("checkpoints", exist_ok=True)
        path = f"checkpoints/{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        mlflow.log_artifact(path)


#  MAIN 

@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="config"
)
def main(cfg):
    setup_mlflow(cfg)
    df = download_data(cfg.infer.path_in_dvc)
    train_df, val_df, test_df = split_by_year(df, cfg)

    train_df, scaler, ohe = preprocess(train_df, cfg, fit=True)
    val_df, _, _ = preprocess(val_df, cfg, scaler, ohe, fit=False)
    test_df, _, _ = preprocess(test_df, cfg, scaler, ohe, fit=False)

    logger = MLflowLogger(cfg.mlflow.experiment_name)

    models = {
        "last_value": LastValueModel(cfg.preprocessing.target_col),
        "ma_3": MovingAverageModel(cfg.preprocessing.target_col, 3),
        "ma_6": MovingAverageModel(cfg.preprocessing.target_col, 6),
        "ma_12": MovingAverageModel(cfg.preprocessing.target_col, 12),
        "seasonal_naive": SeasonalNaiveModel(cfg.preprocessing.target_col),
        "sarima": SARIMAModel(cfg.preprocessing.target_col),
        "xgboost": XGBoostModel(cfg.preprocessing.target_col),
    }

    for name, model in models.items():
        if name == 'xgboost':
            train_df = train_df.drop(cfg.preprocessing.date_features.date_col, axis=1)
            val_df = val_df.drop(cfg.preprocessing.date_features.date_col, axis=1)
            test_df = test_df.drop(cfg.preprocessing.date_features.date_col, axis=1)
            

        print(f"\n===== {name} =====")

        with logger.start_run(name):

            logger.log_params({
                "model": name,
                "train": "2018-2022",
                "val": "2023",
                "test": "2024"
            })

            model.fit(train_df)

            train_pred = model.predict(train_df)
            val_pred = model.predict(val_df)
            test_pred = model.predict(test_df)

            logger.log_metrics(compute_metrics(train_df[cfg.preprocessing.target_col], train_pred), "train_")
            logger.log_metrics(compute_metrics(val_df[cfg.preprocessing.target_col], val_pred), "val_")
            logger.log_metrics(compute_metrics(test_df[cfg.preprocessing.target_col], test_pred), "test_")

            logger.log_model_checkpoint(model, name)


if __name__ == "__main__":
    main()