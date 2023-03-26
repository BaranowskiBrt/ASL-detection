from typing import Optional

import torch
from torch import nn


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        batch_norm: Optional[int] = None,
        activation: Optional[nn.Module] = nn.ReLU(),
        dropout_p: float = 0,
    ) -> None:
        # Setting object as a default is not a good idea, but it doesn't matter for ReLU
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        layers = [nn.Linear(in_features, out_features)]

        if batch_norm:
            layers.append(torch.nn.BatchNorm1d(out_features))
        if activation:
            layers.append(activation)
        if dropout_p:
            layers.append(nn.Dropout(dropout_p))
        self.module = nn.Sequential(*layers)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class LinearModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        keypoints_len: int,
        frame_len,
        dim_no: int = 3,
        dropout_p: float = 0,
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            LinearBlock(
                frame_len * keypoints_len * dim_no, 1024, batch_norm=True, dropout_p=dropout_p
            ),
            LinearBlock(
                frame_len * keypoints_len * dim_no, 1024, batch_norm=True, dropout_p=dropout_p
            ),
            LinearBlock(1024, 512, batch_norm=True, dropout_p=dropout_p),
            LinearBlock(512, 400, batch_norm=True, dropout_p=dropout_p),
            LinearBlock(400, 256, batch_norm=True, dropout_p=dropout_p),
            LinearBlock(256, num_classes, activation=None),
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class LinearSplitModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        keypoints_len: int,
        frame_len,
        dim_no: int = 3,
        dropout_p: float = 0,
    ):
        super().__init__()

        hand_size = frame_len * 21 * dim_no
        body_size = frame_len * (keypoints_len - 2 * 21) * dim_no

        self.hand_model = nn.Sequential(
            nn.Flatten(),
            LinearBlock(hand_size, 512, batch_norm=True, dropout_p=dropout_p),
            LinearBlock(512, 256, batch_norm=True, dropout_p=dropout_p),
            LinearBlock(256, 128, batch_norm=True, dropout_p=dropout_p),
        )
        self.body_model = nn.Sequential(
            nn.Flatten(),
            LinearBlock(body_size, 256, batch_norm=True, dropout_p=dropout_p),
            LinearBlock(256, 128, batch_norm=True, dropout_p=dropout_p),
            LinearBlock(128, 64, batch_norm=True, dropout_p=dropout_p),
        )
        self.head = nn.Sequential(
            LinearBlock(
                2 * self.hand_model[-1].out_features + self.body_model[-1].out_features,
                512,
                batch_norm=True,
                dropout_p=dropout_p,
            ),
            LinearBlock(512, 512, batch_norm=True, dropout_p=dropout_p),
            LinearBlock(512, 256, batch_norm=True, dropout_p=dropout_p),
            LinearBlock(256, num_classes, activation=None),
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Hands are the last 42 keypoints
        hand1 = self.hand_model(x[:, :, -21:, :])
        hand2 = x[:, :, -42:-21, :]
        hand2[:, :, :, 0] = -hand2[:, :, :, 0]
        hand2 = self.hand_model(hand2)
        body = self.body_model(x[:, :, :-42, :])

        return self.head(torch.cat([hand1, hand2, body], dim=1))
