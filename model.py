from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics import Accuracy

DEFAULT_HIDDEN_SIZES = [256, 128, 250]


class AslModel(pl.LightningModule):
    def __init__(self, num_classes: int, rows_per_frame: int, dim_no: int = 3):
        super().__init__()

        self.num_classes = num_classes
        self.rows_per_frame = rows_per_frame

        layers = [nn.Flatten()]
        sizes = [10 * rows_per_frame * dim_no] + DEFAULT_HIDDEN_SIZES + [num_classes]
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i != len(sizes) - 2:
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.valid_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> nn.Sequential:
        print(type(x))
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_no) -> torch.Tensor:
        x, y = batch
        pred = self.forward(x)
        loss = F.cross_entropy(pred, y)
        acc = self.train_accuracy(pred, y)

        self.log("train_accuracy", self.train_accuracy, on_step=True, on_epoch=True)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_no):
        x, y = batch
        pred = self.forward(x)
        self.valid_accuracy(pred, y)
        self.log("valid_accuracy", self.valid_accuracy, on_step=True, on_epoch=True)
        return super().validation_step(batch, batch_no)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
