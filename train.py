import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from datamodule import AslDataModule
from model import AslModel
from pytorch_lightning.loggers import WandbLogger

SEED = 0
BATCH_SIZE = 512
NUM_CLASSES = 250
ROWS_PER_FRAME = 543
INPUT_DIR = "../asl-signs"
ROOT_DIR = "../asl_logging"

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_dir", default=INPUT_DIR)
    parser.add_argument("-d", "--root_dir", default=ROOT_DIR)
    args = parser.parse_args()
    os.makedirs(args.root_dir, exist_ok=True)

    pl.seed_everything(SEED)

    data_module = AslDataModule(
        args.input_dir, rows_per_frame=ROWS_PER_FRAME, seed=SEED, batch_size=BATCH_SIZE
    )
    model = AslModel(num_classes=NUM_CLASSES, rows_per_frame=ROWS_PER_FRAME)
    wandb_logger = WandbLogger(project="ASL", save_dir=args.root_dir)

    trainer = pl.Trainer(logger=wandb_logger, default_root_dir=args.root_dir)
    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())
