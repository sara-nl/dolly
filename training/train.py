import subprocess
import os
from logging import getLogger

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import call, instantiate
from omegaconf import OmegaConf

import time

logger = getLogger(__name__)


@hydra.main(config_path="../conf", version_base=None, config_name="dolly")
def main(experiment):

    logger.info(experiment)

    trainer: pl.Trainer = instantiate(experiment.trainer)
    model: pl.LightningModule = instantiate(experiment.model)
    # train_datamodule: pl.LightningDataModule = instantiate(experiment.train_datamodule)

    # we pass the dataloaders explicitely so we can use memory_format=torch.channels
    # trainer.fit(model, train_datamodule.train_dataloader(), train_datamodule.val_dataloader())

    # return trainer.callback_metrics['val_recon_loss'].item()
    print("Succes!")


if __name__ == '__main__':
    # from utils.conf_helpers import add_resolvers
    # add_resolvers()

    main()