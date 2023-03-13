import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset


def load_relevant_data_subset(pq_path, rows_per_frame):
    data_columns = ["x", "y", "z"]
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / rows_per_frame)
    data = data.values.reshape(n_frames, rows_per_frame, len(data_columns))
    return data.astype(np.float32)


class DataTransform(Module):
    def __init__(self, frame_len):
        super().__init__()
        self.frame_len = frame_len

    def forward(self, x):
        x = x.unsqueeze(0).permute(0, 2, 1, 3)
        x = F.interpolate(x, [self.frame_len, x.shape[-1]], mode="bilinear")
        x = x.permute(0, 2, 1, 3)
        # torch.nan_to_num is not converted properly to tf lite
        x = torch.where(torch.isnan(x), torch.zeros((1)), x)
        return x


class AslDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        rows_per_frame: int,
        sign_dict: Dict[str, int],
        input_dir: str,
        frame_len: int,
    ):
        super().__init__()
        self.df = df
        self.rows_per_frame = rows_per_frame
        self.sign_dict = sign_dict
        self.input_dir = input_dir
        self.frame_len = frame_len
        self.data_transform = DataTransform(self.frame_len)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        x = load_relevant_data_subset(
            self.input_dir / self.df.iloc[index]["path"], self.rows_per_frame
        )

        return self.data_transform(torch.Tensor(x)), self.sign_dict[self.df.iloc[index]["sign"]]

    def __len__(self) -> int:
        return len(self.df)


class AslDataModule(LightningDataModule):
    def __init__(
        self,
        input_dir: Union[Path, str],
        rows_per_frame: int,
        frame_len,
        df_path: str = "train.csv",
        sign_json_path: str = "sign_to_prediction_index_map.json",
        seed: int = 0,
        batch_size: int = 32,
    ):
        super().__init__()

        self.input_dir = Path(input_dir)
        df_path = self.input_dir / "train.csv"
        sign_json_path = self.input_dir / sign_json_path

        self.df_path = df_path
        self.df = pd.read_csv(df_path)
        # self.df = self.df[self.df["sign"] == "TV"]
        # self.df = self.df[:200]
        self.rows_per_frame = rows_per_frame
        self.frame_len = frame_len
        self.sign_json_path = sign_json_path
        with open(self.sign_json_path, "r") as f:
            self.sign_dict = json.load(f)

        self.seed = seed

        self.datasets: Dict[str, Dataset] = {}
        common_overrides = {"batch_size": batch_size, "num_workers": 4}
        self.train_overrides = {"shuffle": True, **common_overrides}
        self.eval_overrides = {"shuffle": False, **common_overrides}
        self.setup()

    def setup(self, stage: Optional[str] = None):
        train_df = self.df.sample(frac=0.8, random_state=self.seed)
        val_df = self.df.drop(train_df.index)
        self.datasets["train"] = AslDataset(
            train_df, self.rows_per_frame, self.sign_dict, self.input_dir, self.frame_len
        )
        self.datasets["val"] = AslDataset(
            val_df, self.rows_per_frame, self.sign_dict, self.input_dir, self.frame_len
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.datasets["train"], **self.train_overrides)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.datasets["val"], **self.eval_overrides)
