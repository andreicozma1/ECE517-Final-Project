import os
from collections import OrderedDict, deque, namedtuple
from typing import Iterator, List, Tuple

import argparse
import gym
import numpy as np
import pandas as pd
import seaborn as sn
from datetime import datetime
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
import matplotlib.pyplot as plt
import pytorch_lightning as pl
# from pl_bolts.models.rl import AdvantageActorCritic
import torch
# from pl_bolts.models.rl.ppo_model import PPO
from rllib.advantage_actor_critic_model import AdvantageActorCritic
from rllib.ppo_model import PPO
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer


def train(args):
    # setup model
    if args.model == "ppo":
        model = PPO(args.env)
    elif args.model == "a2c":
        model = AdvantageActorCritic(args.env)
    else:
        raise ValueError("Model not supported")

    # setup logger
    if args.wandb_project == "":
        logger = CSVLogger(save_dir=args.log_dir, name=args.model)
    else:
        logger = WandbLogger(project=args.wandb_project)

    # setup checkpoints
    # callbacks = []
    # if args.checkpoint_dir != "":
    #     checkpoint_callback = ModelCheckpoint(
    #             dirpath=args.checkpoint_dir,
    #             every_n_epochs=args.checkpoint_freq,
    #             verbose=True
    #         )
    #     callbacks.append(checkpoint_callback)

    # setup trainer
    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=args.max_epochs,
        val_check_interval=args.val_check_interval,
        logger=logger,
        enable_checkpointing=True,
        # callbacks=callbacks
    )
    # train
    trainer.fit(model)
    save_path = os.path.join(args.checkpoint_dir, args.model, datetime.now().isoformat()+".pt")
    print(save_path)
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="LunarLander-v2")
    parser.add_argument("--model", type=str, default="ppo")

    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--val_check_interval", type=int, default=50)

    parser.add_argument("--log_dir", type=str, default="logs/")
    parser.add_argument("--wandb_project", type=str, default="")

    parser.add_argument("--checkpoint_dir", type=str, default="")
    args = parser.parse_args()

    train(args)

