import os

import argparse
import gym
import json
import hashlib
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


def train(i, args):
    # setup model
    if args.model_name == "ppo":
        model = PPO(args.env)
    elif args.model_name == "a2c":
        model = AdvantageActorCritic(args.env)
    else:
        raise ValueError("Model not supported")

    m_hash = hashlib.md5(str(model).encode('utf-8')).hexdigest()
    m_name = f"{m_hash}-{i}"
    # setup logger
    if args.wandb_project == "":
        logger = CSVLogger(save_dir=args.log_dir, name=m_name)
    else:
        logger = WandbLogger(project=args.wandb_project, name=m_name,
                             group=args.model_name, tags=["train"])

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
            max_epochs=args.num_epochs,
            val_check_interval=args.val_check_interval,
            logger=logger,
            enable_checkpointing=True,
            auto_lr_find=True,
            auto_scale_batch_size=True,
            enable_model_summary=True,
            precision=16,
            # callbacks=callbacks
    )
    # train
    trainer.fit(model)
    model_save_dir = os.path.join(args.checkpoint_dir, args.model_name)
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_dir = os.path.join(model_save_dir, args.env)
    os.makedirs(model_save_dir, exist_ok=True)
    save_path = os.path.join(model_save_dir, m_name + ".pt")
    print(save_path)
    torch.save(model.state_dict(), save_path)


def main(args):
    for i in range(args.num_models):
        train(i, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", type=str, default="LunarLander-v2")
    parser.add_argument("-nm", "--num_models", type=int, default=10)
    parser.add_argument("-m", "--model_name", type=str, default="ppo")
    parser.add_argument("-ne", "--num_epochs", type=int, default=150)
    parser.add_argument("--val_check_interval", type=int, default=25)

    parser.add_argument("--log_dir", type=str, default="logs/")
    parser.add_argument("--wandb_project", type=str, default="rl_project")

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/")

    args = parser.parse_args()
    main(args)
