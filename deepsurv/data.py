from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class SurvivalDataSpec:
    feature_columns: list
    duration_column: str
    event_column: str
    id_column: str | None = None
    weight_column: str | None = None


class SurvivalDataset(Dataset):
    def __init__(self, dataframe, spec):
        self.spec = spec
        self.frame = dataframe.reset_index(drop=True).copy()

        missing = [
            column
            for column in [spec.duration_column, spec.event_column, *spec.feature_columns]
            if column not in self.frame.columns
        ]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        columns = [*spec.feature_columns, spec.duration_column, spec.event_column]
        if spec.id_column is not None:
            columns.append(spec.id_column)
        if spec.weight_column is not None:
            columns.append(spec.weight_column)

        self.frame = self.frame[columns].dropna().reset_index(drop=True)
        self.features = self.frame[spec.feature_columns].to_numpy(dtype=np.float32)
        self.durations = self.frame[spec.duration_column].to_numpy(dtype=np.float32)
        self.events = self.frame[spec.event_column].to_numpy(dtype=np.float32)
        self.weights = None
        if spec.weight_column is not None:
            self.weights = self.frame[spec.weight_column].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        item = {
            "x": torch.tensor(self.features[index], dtype=torch.float32),
            "time": torch.tensor(self.durations[index], dtype=torch.float32),
            "event": torch.tensor(self.events[index], dtype=torch.float32),
        }
        if self.weights is not None:
            item["weight"] = torch.tensor(self.weights[index], dtype=torch.float32)
        return item


def load_survival_dataframe(csv_path):
    df = pd.read_csv(csv_path)
    return df.loc[:, ~df.columns.str.contains("^Unnamed")]
