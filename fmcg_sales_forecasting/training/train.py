import os
import subprocess

import dvc.api
import hydra
import joblib
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

from fmcg_sales_forecasting.dataset.fmcg_forecast_dataset import FMCGForecastDataset

from ..model.tft_model import TFTModel, mae, rmse, smape, wape
from ..preprocessing.preprocess import preprocess_fmcg


class TFTLightningModule(pl.LightningModule):
    def __init__(self, cfg, num_vars: int):
        super().__init__()

        self.save_hyperparameters(ignore=["num_vars"])

        self.model = TFTModel(
            cfg=cfg.model,
            num_vars=num_vars
        )
        self.lr = cfg.model.lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)
        loss = mae(y, y_hat)
        self.log("train_mae", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)
        self.log("val_mae", mae(y, y_hat), prog_bar=True)
        self.log("val_rmse", rmse(y, y_hat))
        self.log("val_smape", smape(y, y_hat))
        self.log("val_wape", wape(y, y_hat))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def download_data(path_in_dvc: str) -> pd.DataFrame:
    with dvc.api.open(path_in_dvc, repo=".") as f:
        df = pd.read_csv(f)
    return df


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    torch.manual_seed(42)

    df = download_data("data/raw/data.csv")

    df, scaler, ohe = preprocess_fmcg(
        df,
        cfg=cfg.preprocessing,
        fit=True
    )

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(scaler, "artifacts/scaler.pkl")
    joblib.dump(ohe, "artifacts/ohe.pkl")

    train_dataset = FMCGForecastDataset(
        df,
        past_window=cfg.dataset.past_window,
        horizon=cfg.dataset.horizon,
        year_col=cfg.dataset.year_col,
        year_max=2022
    )

    val_dataset = FMCGForecastDataset(
        df,
        past_window=cfg.dataset.past_window,
        horizon=cfg.dataset.horizon,
        year_col=cfg.dataset.year_col,
        year_min=2023,
        year_max=2023
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=4
    )

    num_vars = len(train_dataset.feature_cols)
    model = TFTLightningModule(cfg, num_vars=num_vars)


    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg.train.checkpoint_dir,
        filename="best",
        monitor="val_mae",
        save_top_k=3,
        mode="min",
    )

    early_stop_cb = EarlyStopping(
        monitor="val_mae",
        patience=cfg.train.patience,
        mode="min"
    )

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri
    )

    git_commit = subprocess.getoutput("git rev-parse HEAD")
    hparams = OmegaConf.to_container(cfg, resolve=True)
    hparams["git_commit"] = git_commit
    mlflow_logger.log_hyperparams(hparams)

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accel,
        devices=cfg.train.devices,
        callbacks=[checkpoint_cb, early_stop_cb],
        log_every_n_steps=1,
        deterministic=True,
        logger=mlflow_logger
    )

    trainer.fit(model, train_loader, val_loader)
    print(f"Лучшие чекпоинты сохранены в {cfg.train.checkpoint_dir}")


if __name__ == "__main__":
    main()
