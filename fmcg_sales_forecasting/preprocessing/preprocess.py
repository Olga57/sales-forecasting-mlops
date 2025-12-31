import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def fill_missing_group_mean(df: pd.DataFrame, group_col: str):
    for col in df.columns:
        if df[col].isna().any() and col != group_col:
            df[col] = df.groupby(group_col)[col].transform(lambda x: x.fillna(x.mean()))
    return df


def add_date_features(df: pd.DataFrame, cfg: DictConfig):
    date_col = cfg.date_features.date_col
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    if cfg.date_features.add_year:
        df["Year"] = df[date_col].dt.year

    if cfg.date_features.add_month_sin_cos:
        month = df[date_col].dt.month
        df["Month_sin"] = np.sin(2 * np.pi * month / 12)
        df["Month_cos"] = np.cos(2 * np.pi * month / 12)

    return df


def scale_numeric(df: pd.DataFrame, numeric_cols: list[str],
                   fit: bool, scaler: StandardScaler | None):
    if fit:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df, scaler


def encode_categorical(df: pd.DataFrame, categorical_cols: list[str],
                        fit: bool, ohe: OneHotEncoder | None, cfg: DictConfig):
    if categorical_cols:
        if fit:
            ohe = OneHotEncoder(
                sparse_output=cfg.categorical.ohe.sparse,
                drop=cfg.categorical.ohe.drop,
                handle_unknown=cfg.categorical.ohe.handle_unknown,
            )
            cat_data = ohe.fit_transform(df[categorical_cols])
        else:
            cat_data = ohe.transform(df[categorical_cols])

        cat_df = pd.DataFrame(
            cat_data,
            columns=ohe.get_feature_names_out(categorical_cols),
            index=df.index,
        )
        df = pd.concat([df.drop(columns=categorical_cols), cat_df], axis=1)

    return df, ohe


def preprocess_fmcg(
    df: pd.DataFrame,
    cfg: DictConfig,
    scaler: StandardScaler | None = None,
    ohe: OneHotEncoder | None = None,
    fit: bool = True,
):
    df = df.copy()

    target_col = cfg.target_col
    row_label_col = cfg.row_label_col

    # Drop columns with too many missing values
    thresh = cfg.missing_values.drop_threshold_ratio * len(df)
    df = df.loc[:, df.isna().sum() <= thresh]

    # Fill missing values
    if cfg.missing_values.fill_strategy == "group_mean":
        df = fill_missing_group_mean(df, row_label_col)

    # Date features
    if cfg.date_features.enabled and cfg.date_features.date_col in df.columns:
        df = add_date_features(df, cfg)

    # Target transformation
    if cfg.target_transform.log_transform:
        df[target_col] = np.log1p(df[target_col])

    # Numeric scaling
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [
        c for c in numeric_cols if c not in [target_col] + cfg.scaling.exclude_cols
    ]
    if cfg.scaling.enabled:
        df, scaler = scale_numeric(df, numeric_cols, fit, scaler)

    # Categorical encoding
    if cfg.categorical.enabled:
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        categorical_cols = [
            c for c in categorical_cols if c not in cfg.categorical.drop_cols
        ]
        df, ohe = encode_categorical(df, categorical_cols, fit, ohe, cfg)

    return df, scaler, ohe
