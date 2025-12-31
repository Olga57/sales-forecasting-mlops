import pandas as pd
import torch
from torch.utils.data import Dataset


class FMCGForecastDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        target_col="MT_Volume KG",
        time_col="PER_SDESC",
        past_window=3,
        horizon=1,
        year_col="Year",
        year_min=None,
        year_max=None,
    ):
        self.target_col = target_col
        self.time_col = time_col
        self.past_window = past_window
        self.horizon = horizon
        self.year_col = year_col
        self.year_min = year_min
        self.year_max = year_max

        df = df.copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        self.df = df

        self.feature_cols = [c for c in df.columns if c not in [target_col, time_col]]

        self.samples = self.create_sequences()

        if self.year_col is not None and self.year_col in df.columns:
            filtered_samples = []
            for x_seq, y_seq, target_dates in self.samples:
                target_year = pd.to_datetime(target_dates[-1]).year
                if ((self.year_min is None or target_year >= self.year_min) and
                    (self.year_max is None or target_year <= self.year_max)):
                    filtered_samples.append((x_seq, y_seq, target_dates))
            self.samples = filtered_samples

        drop_cols = [col for col in ["Year", "PER_SDESC"] if col in self.df.columns]
        self.df = self.df.drop(columns=drop_cols)
        self.feature_cols = [c for c in self.feature_cols if c in self.df.columns]

    def create_sequences(self):
        sequences = []
        company_starts = [0]
        dates = self.df[self.time_col].values
        for i in range(1, len(dates)):
            if dates[i] < dates[i - 1]:
                company_starts.append(i)
        company_starts.append(len(dates))

        x_values = self.df[self.feature_cols].values.astype("float32")
        y_values = self.df[self.target_col].values.astype("float32")
        date_values = self.df[self.time_col].values

        for start, end in zip(company_starts[:-1], company_starts[1:], strict=True):
            n_steps = end - start
            for i in range(n_steps - self.past_window - self.horizon + 1):
                x_seq = x_values[start + i : start + i + self.past_window]
                x_seq = torch.tensor(x_seq, dtype=torch.float32).unsqueeze(-1)

                y_seq = y_values[
                    start + i : start + i + self.past_window + self.horizon
                ][-self.horizon :]
                y_seq = torch.tensor(y_seq, dtype=torch.float32)

                target_dates_seq = date_values[
                    start + i : start + i + self.past_window + self.horizon
                ][-self.horizon :]
                sequences.append((x_seq, y_seq, target_dates_seq))

        return sequences

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        features, target, _ = self.samples[idx]
        return features, target
