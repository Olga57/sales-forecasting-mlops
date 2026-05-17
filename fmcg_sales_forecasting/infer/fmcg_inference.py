import subprocess
import dvc.api
import hydra
import joblib
import pandas as pd
import torch

from omegaconf import DictConfig
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

from ..dataset.fmcg_forecast_dataset import FMCGForecastDataset
from ..model.tft_model import mae, rmse, smape, wape
from ..preprocessing.preprocess import preprocess_fmcg
from ..training.train import TFTLightningModule


def download_data(path_in_dvc: str) -> pd.DataFrame:
    with dvc.api.open(path_in_dvc, repo=".") as f:
        return pd.read_csv(f)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    torch.manual_seed(42)

    # ======================
    # LOGGING
    # ======================
    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri
    )

    mlflow_logger.log_hyperparams({
        "git_commit": subprocess.getoutput("git rev-parse HEAD")
    })

    # ======================
    # LOAD DATA
    # ======================
    df = download_data(cfg.infer.path_in_dvc)

    scaler = joblib.load("artifacts/scaler.pkl")
    ohe = joblib.load("artifacts/ohe.pkl")

    df, _, _ = preprocess_fmcg(
        df,
        ohe=ohe,
        scaler=scaler,
        cfg=cfg.preprocessing,
        fit=False
    )

    # ======================
    # DATASET (SOURCE OF TRUTH)
    # ======================
    test_dataset = FMCGForecastDataset(
        df,
        past_window=cfg.dataset.past_window,
        horizon=cfg.dataset.horizon,
        year_col=cfg.dataset.year_col,
        year_min=cfg.infer.year_min,
        year_max=cfg.infer.year_max
    )

    num_vars = len(test_dataset.feature_cols)

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.infer.num_workers
    )


    ckpt = torch.load(
        cfg.infer.checkpoint_path,
        map_location="cpu",
        weights_only=False
    )


    model = TFTLightningModule(
        cfg=cfg,
        num_vars=num_vars
    )

    model.load_state_dict(ckpt["state_dict"])
    model.eval()


    all_preds, all_targets = [], []

    with torch.no_grad():
        for x, y in test_loader:
            y_hat, _ = model(x)
            all_preds.append(y_hat)
            all_targets.append(y)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)


    metrics = {
        "MAE": mae(all_targets, all_preds).item(),
        "RMSE": rmse(all_targets, all_preds).item(),
        "sMAPE": smape(all_targets, all_preds).item(),
        "WAPE": wape(all_targets, all_preds).item()
    }

    for k, v in metrics.items():
        print(f"{k}: {v}")

    mlflow_logger.log_metrics(metrics)


if __name__ == "__main__":
    main()