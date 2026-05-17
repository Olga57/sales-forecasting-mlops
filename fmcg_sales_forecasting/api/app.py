from fastapi import FastAPI
import torch
import joblib
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from dataclasses import dataclass
from fmcg_sales_forecasting.baseline.run_baseline import preprocess
import dvc.api
from fmcg_sales_forecasting.preprocessing.preprocess import preprocess_fmcg
from fmcg_sales_forecasting.dataset.fmcg_forecast_dataset import FMCGForecastDataset
from fmcg_sales_forecasting.model.tft_model import mae as mae_torch, rmse as rmse_torch, smape as smape_torch, wape as wape_torch
from fmcg_sales_forecasting.training.train import TFTLightningModule

from fmcg_sales_forecasting.baseline.run_baseline import mae, rmse, smape, wape

from hydra import compose, initialize_config_dir


app = FastAPI(title="FMCG Eval API")

BASE_DIR = Path("/app")

@dataclass
class AppState:
    cfg: any = None
    scaler: any = None
    ohe: any = None
    df_raw: pd.DataFrame = None
    base_models: dict = None


state = AppState()


def download_data(path_in_dvc: str) -> pd.DataFrame:
    with dvc.api.open(path_in_dvc, repo=str(BASE_DIR)) as f:
        return pd.read_csv(f)


def load_config():
    config_dir = BASE_DIR / "fmcg_sales_forecasting" / "configs"

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config")

    return cfg


def load_base_models():
    models_dir = BASE_DIR / "checkpoints"

    models = {}
    for p in models_dir.glob("*.pkl"):
        models[p.stem] = joblib.load(p)

    return models


@app.on_event("startup")
def startup():
    state.cfg = load_config()

    state.scaler = joblib.load(BASE_DIR / "artifacts" / "scaler.pkl")
    state.ohe = joblib.load(BASE_DIR /"artifacts" / "ohe.pkl")

    state.df_raw = download_data(state.cfg.infer.path_in_dvc)

    state.base_models = load_base_models()

@app.get("/predict")
def predict(model_name: str, model_file: str):
    cfg = state.cfg
    scaler = state.scaler
    ohe = state.ohe
    df_raw = state.df_raw

    checkpoint_path = BASE_DIR /"checkpoints" / model_file

    if not checkpoint_path.exists():
        return {"error": f"Model not found: {model_file}"}


    df_raw = df_raw.copy()
    date_col = cfg.preprocessing.date_features.date_col

    df_raw[date_col] = pd.to_datetime(df_raw[date_col], errors="coerce")
# сделал с 2023 val+test можно поменять 
    df_filtered = df_raw[
        (df_raw[date_col].notna()) &
        (df_raw[date_col] >= pd.Timestamp("2023-01-01"))
    ]
    target_col = cfg.preprocessing.target_col

  
    if model_name == "base":

        df, _, _ = preprocess(
            df_filtered,
            cfg=cfg,
            scaler=scaler,
            ohe=ohe,
            fit=False
        )

        model = joblib.load(checkpoint_path)

        
        if model_file == "xgboost.pkl":
            df = df.drop(columns=[date_col])

        preds = model.predict(df)

        if target_col in df.columns:
            y_true = df[target_col].values
            preds = np.array(preds)

            return {
                "model": model_name,
                "model_file": model_file,
                "MAE": float(mae(y_true, preds)),
                "RMSE": float(rmse(y_true, preds)),
                "sMAPE": float(smape(y_true, preds)),
                "WAPE": float(wape(y_true, preds)),
            }

        return {
            "model": model_name,
            "model_file": model_file,
            "predictions": preds.tolist()
        }


    if model_name == "tft":

        df, _, _ = preprocess_fmcg(
            df_filtered,
            cfg=cfg.preprocessing,
            scaler=scaler,
            ohe=ohe,
            fit=False
        )

        dataset = FMCGForecastDataset(
            df,
            past_window=cfg.dataset.past_window,
            horizon=cfg.dataset.horizon,
        )

        loader = DataLoader(
            dataset,
            batch_size=cfg.dataset.batch_size,
            shuffle=False
        )

        model = TFTLightningModule.load_from_checkpoint(
            checkpoint_path=str(checkpoint_path),
            cfg=cfg,
            num_vars=len(dataset.feature_cols),
            map_location="cpu",
            weights_only=False
        )

        model.eval()

        preds, targets = [], []

        with torch.no_grad():
            for x, y in loader:
                y_hat, _ = model(x)
                preds.append(y_hat)
                targets.append(y)

        if len(preds) == 0:
            return {
                "error": "TFT produced no batches",
                "dataset_len": len(dataset),
                "df_shape": df.shape
            } 

        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)

        return {
            "model": model_name,
            "model_file": model_file,
            "MAE": float(mae_torch(targets, preds)),
            "RMSE": float(rmse_torch(targets, preds)),
            "sMAPE": float(smape_torch(targets, preds)),
            "WAPE": float(wape_torch(targets, preds)),
        }

    return {"error": f"Unknown model_name: {model_name}"}