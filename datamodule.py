import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule
from torch.nn import Identity, Module
from torch.utils.data import DataLoader, Dataset

from augment import DataAugmenter

ROWS_PER_FRAME = 543


def load_relevant_data_subset(pq_path):
    data_columns = ["x", "y", "z"]
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


class DataTransform(Module):
    def __init__(
        self, frame_len, keypoints, mean=None, std=None, interpolate=True, interpolate_mode="linear"
    ):
        super().__init__()
        self.frame_len = frame_len
        self.keypoints = keypoints
        self.mean = torch.Tensor(mean or (0.471518, 0.460818, -0.045338))
        self.std = torch.Tensor(std or (0.103356, 0.240428, 0.302280))
        self.interpolate = interpolate
        self.interpolate_mode = interpolate_mode

    def forward(self, x):
        x = x[:, self.keypoints, :]
        x = (x - self.mean) / self.std
        if self.interpolate:
            x = x.permute(1, 2, 0)
            x = F.interpolate(x, [self.frame_len], mode=self.interpolate_mode)
            x = x.permute(2, 0, 1)
        else:
            x = x[: self.frame_len, :, :]
            pad_len = self.frame_len - x.shape[0]
            # Pad with zeros at the end of the third dimension from the end
            x = F.pad(input=x, pad=(0, 0, 0, 0, 0, pad_len), mode="constant", value=0)
        # torch.nan_to_num is not converted properly to tf lite
        x = torch.where(torch.isnan(x), torch.zeros((1)), x).unsqueeze(0)

        return x


class AslDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        keypoints: List[int],
        sign_dict: Dict[str, int],
        input_dir: str,
        frame_len: int,
        augmenter_cfg: Optional[Dict] = None,
        interpolate: bool = True,
        frame_dropout_p: float = 0,
    ):
        super().__init__()
        self.df = df
        self.keypoints = keypoints
        self.sign_dict = sign_dict
        self.input_dir = input_dir
        self.frame_len = frame_len
        self.data_transform = DataTransform(self.frame_len, self.keypoints, interpolate=interpolate)
        self.augmentations = DataAugmenter(augmenter_cfg) if augmenter_cfg else Identity()
        self.frame_dropout_p = frame_dropout_p

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.input_dir / self.df.iloc[index]["path"]
        if str(path).endswith(".pkl"):
            with open(path, "rb") as f:
                x = pickle.load(f)
        else:
            x = load_relevant_data_subset(path)
            if self.frame_dropout_p:
                passed_frames = torch.rand(len(x)) > self.frame_dropout_p
                if passed_frames.any():
                    x = x[passed_frames]
            x = self.data_transform(torch.Tensor(x))

        x = x.squeeze(0)
        x = self.augmentations(x)
        return x, self.sign_dict[self.df.iloc[index]["sign"]]

    def __len__(self) -> int:
        return len(self.df)


class AslDataModule(LightningDataModule):
    def __init__(
        self,
        input_dir: Union[Path, str],
        keypoints: List[int],
        frame_len,
        augmenter_cfg: Dict,
        df_path: str = "train.csv",
        sign_json_path: str = "sign_to_prediction_index_map.json",
        seed: int = 0,
        batch_size: int = 32,
        num_workers: int = 4,
        train_frac: float = 0.9,
        signer_split=True,
        interpolate: bool = True,
    ):
        super().__init__()
        self.input_dir = Path(input_dir)
        df_path = self.input_dir / "train.csv"
        sign_json_path = self.input_dir / sign_json_path

        self.df_path = df_path
        self.df = pd.read_csv(df_path)
        # self.df = self.df[self.df["sign"] == "TV"]
        # self.df = self.df[:200]
        self.keypoints = keypoints
        self.frame_len = frame_len
        self.sign_json_path = sign_json_path
        with open(self.sign_json_path, "r") as f:
            self.sign_dict = json.load(f)

        self.seed = seed

        self.datasets: Dict[str, Dataset] = {}
        common_overrides = {"batch_size": batch_size, "num_workers": num_workers}
        self.train_overrides = {"shuffle": True, **common_overrides}
        self.eval_overrides = {"shuffle": False, **common_overrides}

        if signer_split:
            train_df, val_df = self.split_by_signer(train_frac)
        else:
            train_df = self.df.sample(frac=train_frac, random_state=self.seed)
            val_df = self.df.drop(train_df.index)
        self.datasets["train"] = AslDataset(
            train_df,
            self.keypoints,
            self.sign_dict,
            self.input_dir,
            self.frame_len,
            augmenter_cfg=augmenter_cfg,
            interpolate=interpolate,
        )
        self.datasets["val"] = AslDataset(
            val_df,
            self.keypoints,
            self.sign_dict,
            self.input_dir,
            self.frame_len,
            interpolate=interpolate,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.datasets["train"], **self.train_overrides)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.datasets["val"], **self.eval_overrides)

    def split_by_signer(self, train_frac: float, train_smallest_inst: bool = True):
        max_inst = len(self.df) * train_frac
        train_signer_ids = []

        signer_counts = self.df["participant_id"].value_counts()
        signers = (
            signer_counts[::-1].iteritems() if train_smallest_inst else signer_counts.iteritems()
        )
        run_sum = 0
        for id, no_inst in signers:
            # Round to a signer
            if abs(run_sum + no_inst - max_inst) > abs(run_sum - max_inst):
                break
            run_sum += no_inst
            train_signer_ids.append(id)
        print(
            f"No. train signers: {len(train_signer_ids)}, No. val signers {len(list(signer_counts)) - len(train_signer_ids)}"
        )
        train_df = self.df[self.df["participant_id"].isin(train_signer_ids)]
        val_df = self.df.drop(train_df.index)
        return train_df, val_df
