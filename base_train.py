import os
from collections import OrderedDict, deque, namedtuple
from typing import Iterator, List, Tuple

import gym
import numpy as np
import pandas as pd
import seaborn as sn
from IPython.core.display import display
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import CSVLogger
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pl_bolts.models.rl import AdvantageActorCritic
import torch
from pl_bolts.models.rl.ppo_model import PPO
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

# PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")

ppo = PPO("LunarLander-v2")

wandb_logger = WandbLogger(project="rl-final-proj")

trainer = Trainer(
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    max_epochs=150,
    val_check_interval=50,
    logger=wandb_logger
)

trainer.fit(ppo)

