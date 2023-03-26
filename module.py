from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics import Accuracy


class AslLightningModule(pl.LightningModule):
    def __init__(self, num_classes: int, model: nn.Module):
        super().__init__()

        self.num_classes = num_classes

        self.model = model

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
        self.log("trainer/learning_rate", self.lr_schedulers().get_last_lr()[0])

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_no) -> torch.Tensor:
        x, y = batch
        pred = self.forward(x)
        loss = F.cross_entropy(pred, y)
        self.train_accuracy(pred, y)
        acc_per_class = self.train_accuracy_per_class(pred, y)

        self.log("train_accuracy", self.train_accuracy, on_step=True, on_epoch=True)
        self.log_dict(
            {
                f"per_class_metrics_train/accuracy_{i}": acc
                for i, acc in zip(range(self.num_classes), acc_per_class)
            }
        )

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_no):
        x, y = batch
        pred = self.forward(x)
        self.val_accuracy(pred, y)
        acc_per_class = self.val_accuracy_per_class(pred, y)
        self.log("val_accuracy", self.val_accuracy, on_step=True, on_epoch=True)
        self.log_dict(
            {
                f"per_class_metrics_val/accuracy_{i}": acc
                for i, acc in zip(range(self.num_classes), acc_per_class)
            }
        )
        return super().validation_step(batch, batch_no)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_no):
        x, y = batch
        pred = self.forward(x)
        self.val_accuracy(pred, y)
        acc_per_class = self.val_accuracy_per_class(pred, y)
        self.log("test_accuracy", self.val_accuracy, on_step=True, on_epoch=True)
        self.log_dict(
            {
                f"per_class_metrics_test/accuracy_{i}": acc
                for i, acc in zip(range(self.num_classes), acc_per_class)
            }
        )
        return super().test_step(batch, batch_no)

    def configure_optimizers(
        self,
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
        return [optimizer], [scheduler]
