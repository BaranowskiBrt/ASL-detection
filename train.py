import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from config import load_configs
from datamodule import AslDataModule
from keypoints import extract_keypoints
from model import AslModel

if __name__ == "__main__":
    cfg = load_configs("configs/default.yaml")
    os.makedirs(cfg.root_dir, exist_ok=True)

    pl.seed_everything(cfg.seed)

    extracted_keypoints = extract_keypoints(
        silhouette=cfg.silhouette,
        lips=cfg.lips,
        eyes=cfg.eyes,
        eyebrows=cfg.eyebrows,
        rest_of_face=cfg.rest_of_face,
        pose=cfg.pose,
        hands=cfg.hands,
    )
    print("Number of extracted keypoints:", len(extracted_keypoints))
    data_module = AslDataModule(
        cfg.input_dir,
        keypoints=extracted_keypoints,
        frame_len=cfg.frame_len,
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        train_frac=cfg.train_frac,
    )
    model = AslModel(
        num_classes=cfg.num_classes,
        keypoints_len=len(extracted_keypoints),
        frame_len=cfg.frame_len,
        dropout_p=cfg.dropout_p,
    )
    logger = WandbLogger(project="ASL", save_dir=cfg.root_dir) if cfg.log else None

    callbacks = []
    val_loader = None
    if cfg.train_frac < 1:
        callbacks.append(EarlyStopping(monitor="val_accuracy", patience=20, mode="max"))
        callbacks.append(
            ModelCheckpoint(
                dirpath=cfg.root_dir,
                monitor="val_accuracy",
                mode="max",
                filename="model-{epoch:02d}-{val_accuracy:.2f}",
                every_n_epochs=5,
            )
        )
        val_loader = data_module.val_dataloader()

    print(model)
    print(cfg)
    trainer = pl.Trainer(
        logger=logger, default_root_dir=cfg.root_dir, callbacks=callbacks, max_epochs=100
    )

    trainer.fit(model, data_module.train_dataloader(), val_loader)
    # trainer.test(model, data_module.val_dataloader(), "../model.ckpt")
