from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics import Accuracy

DEFAULT_HIDDEN_SIZES = [512, 256, 128, 64, 32]


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


class AslModel(pl.LightningModule):
    def __init__(
        self, num_classes: int, keypoints_len: int, frame_len, dim_no: int = 3, dropout_p: float = 0
    ):
        super().__init__()

        self.num_classes = num_classes
        self.keypoints_len = keypoints_len
        self.frame_len = frame_len

        self.model = nn.Sequential(
            nn.Flatten(),
            # LinearBlock(frame_len * keypoints_len * dim_no, 1024),
            # LinearBlock(1024, 512),
            # LinearBlock(512, 256),
            # LinearBlock(256, 128, dropout_p=0.3),
            LinearBlock(
                frame_len * keypoints_len * dim_no, 1024, batch_norm=True, dropout_p=dropout_p
            ),
            LinearBlock(1024, 512, batch_norm=True, dropout_p=dropout_p),
            LinearBlock(512, 256, batch_norm=True, dropout_p=dropout_p),
            # LinearBlock(256, 128, batch_norm=True),
            LinearBlock(256, num_classes, activation=None),
        )

        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_accuracy_per_class = Accuracy(
            task="multiclass", num_classes=num_classes, average=None
        )
        self.val_accuracy_per_class = Accuracy(
            task="multiclass", num_classes=num_classes, average=None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def on_train_epoch_start(self):
        self.log("learning_rate", self.lr_schedulers().get_last_lr()[0])

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_no) -> torch.Tensor:
        x, y = batch
        pred = self.forward(x)
        loss = F.cross_entropy(pred, y)
        self.train_accuracy(pred, y)

        self.log("train_accuracy", self.train_accuracy, on_step=True, on_epoch=True)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_no):
        x, y = batch
        pred = self.forward(x)
        self.val_accuracy(pred, y)
        self.val_accuracy_per_class(pred, y)
        self.log("val_accuracy", self.val_accuracy, on_step=True, on_epoch=True)
        self.log("val_accuracy_per_class", self.val_accuracy_per_class, on_step=True, on_epoch=True)
        return super().validation_step(batch, batch_no)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_no):
        x, y = batch
        pred = self.forward(x)
        self.val_accuracy(pred, y)
        self.test_accuracy_per_class(pred, y)
        self.log("test_accuracy", self.val_accuracy, on_step=True, on_epoch=True)
        self.log(
            "test_accuracy_per_class", self.test_accuracy_per_class, on_step=True, on_epoch=True
        )
        return super().test_step(batch, batch_no)

    def configure_optimizers(
        self,
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
        return [optimizer], [scheduler]
        # return optimizer
