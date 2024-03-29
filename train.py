import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from config import load_configs
from datamodule import AslDataModule
from keypoints import extract_keypoints
from models import choose_model
from module import AslLightningModule

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
        left_hand=cfg.hands,
        right_hand=cfg.hands,
    )
    keypoint_len = len(extracted_keypoints)
    print("Number of extracted keypoints:", keypoint_len)
    data_module = AslDataModule(
        cfg.input_dir,
        keypoints=extracted_keypoints,
        frame_len=cfg.frame_len,
        augmenter_cfg=cfg.augmenter,
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        train_frac=cfg.train_frac,
        signer_split=cfg.signer_split,
        interpolate=cfg.model_type in ["linear", "linear_split"],
        # interpolate=True,
    )

    model = choose_model(
        cfg.model_type, cfg.num_classes, keypoint_len, cfg.frame_len, cfg.dropout_p
    )

    lightning_module = AslLightningModule(cfg.num_classes, model)
    logger = (
        WandbLogger(project="ASL", name=cfg.run_name, save_dir=cfg.root_dir) if cfg.log else None
    )

    callbacks = []
    val_loader = None
    if cfg.train_frac < 1:
        callbacks.append(EarlyStopping(monitor="val_accuracy", patience=20, mode="max"))
        callbacks.append(
            ModelCheckpoint(
                dirpath=cfg.root_dir,
                monitor="val_accuracy",
                mode="max",
                filename="model-{epoch:02d}-{val_accuracy:.3f}",
                every_n_epochs=5,
            )
        )
        val_loader = data_module.val_dataloader()

    print(cfg)
    trainer = pl.Trainer(
        logger=logger, default_root_dir=cfg.root_dir, callbacks=callbacks, max_epochs=300
    )

    trainer.fit(lightning_module, data_module.train_dataloader(), val_loader)
    # trainer.test(model, data_module.val_dataloader(), "../model.ckpt")
