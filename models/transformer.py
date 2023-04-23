import math

import torch
from torch import nn

from .linear import LinearBlock


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.pe[:, : x.shape[1], :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        keypoints_len: int,
        dim_no: int = 3,
        max_frame_len: int = 100,
        dropout_p: float = 0,
    ):
        super().__init__()

        flattened_size = keypoints_len * dim_no
        embedding_size = 128
        print(flattened_size)

        self.transformer = nn.Sequential(
            nn.Flatten(2),
            LinearBlock(
                flattened_size,
                embedding_size,
                batch_norm=True,
                is_sequential=True,
                dropout_p=dropout_p,
            ),
            PositionalEncoding(embedding_size, dropout_p, max_frame_len),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=embedding_size, nhead=4, dim_feedforward=max_frame_len
                ),
                num_layers=1,
            ),
        )
        self.head = nn.Sequential(
            LinearBlock(embedding_size, num_classes, activation=None),
        )

    def embedding_mean(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        adjusted_mask = mask.unsqueeze(-1)
        return (x * adjusted_mask).sum(1) / adjusted_mask.sum(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask_padding = ~torch.all(x.view(*x.shape[:2], -1).isclose(torch.Tensor([0])), dim=2)
        x = self.transformer(x)
        x = self.embedding_mean(x, mask_padding)
        return self.head(x)
