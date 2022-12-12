import hashlib
import os
import pprint
import sys
import time

import numpy as np
import torch
from lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from rllib.PPO2 import PPO2
from rllib.PPO1 import PPO1
from rllib.Utils import discount_rewards
from rllib.examples.A2CExample import AdvantageActorCritic
from rllib.examples.PPOExample import PPO


class Model:

    def __init__(self, seed=None):
        self.seed = seed
        if seed is not None:
            print(f"WARNING: Seeding everything with seed: {seed}")
            seed_everything(self.seed)
        self.model = None
        self.name = None
        self.model_hash = None
        self.model_save_dir = None
        self.loggers = []

    def create_model(self, env, model_name, model_params=None, checkpoint_dir='checkpoints'):
        model_params = model_params or {}
        # setup model
        models = {
                "ppo_ex": PPO,
                "a2c_ex": AdvantageActorCritic,
                "ppo_1" : PPO1,
                "ppo_2" : PPO2,
        }
        if model_name not in models:
            print(f"ERROR: Model {model_name} not supported")
            print("Available models:")
            for model in models:
                print(f" - {model}")
            sys.exit(1)
        self.name = model_name
        self.model = models[self.name](env, *model_params)
        self.model_hash = str(hashlib.md5(str(self.model).encode('utf-8')).hexdigest())
        self.model_save_dir = os.path.join(checkpoint_dir, self.name, env)
        print(f"Args:\n{pprint.pformat(self.__dict__, width=30)}")

    def load_model(self, checkpoint_path, model_params=None):
        model_params = model_params or {}
        model_path_split = checkpoint_path.split('/')
        model_name, env = model_path_split[-3], model_path_split[-2]
        self.create_model(env, model_name, model_params)
        self.model.load_state_dict(torch.load(checkpoint_path))
        print(f"Args:\n{pprint.pformat(self.__dict__, width=30)}")

    def train(self, num_epochs, val_check_interval=None):
        trainer = Trainer(
                accelerator="auto",
                devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
                max_epochs=num_epochs,
                val_check_interval=val_check_interval,
                logger=self.loggers,
                enable_checkpointing=True,
                # auto_lr_find=True,
                # auto_scale_batch_size=True,
                precision=32,
        )
        if self.model is None:
            raise ValueError("ERROR: Model hasn't been created/loaded")
        trainer.fit(self.model)
        self.save_model()

    def eval(self):
        if self.model is None:
            raise ValueError("ERROR: Model hasn't been created/loaded")
        done = False
        state = torch.FloatTensor(self.model.env.reset())
        total_rewards, total_steps = 0, 0
        pred_value = []
        hist_rewards = []
        while not done:
            state = state.to(device=self.model.device)
            next_state, value, reward, done = None, None, None, None
            # api for models are a bit different for getting state and value
            if "ppo" in self.name:
                with torch.no_grad():
                    pi, action, value = self.model(state)
                next_state, reward, done, _ = self.model.env.step(action.cpu().numpy())

            elif "a2c" in self.name:
                with torch.no_grad():
                    action = self.model.agent(state, self.model.device)[0]
                    _, value = self.model.net(state.reshape((1, -1)))
                next_state, reward, done, _ = self.model.env.step(action.cpu().numpy())
            else:
                raise ValueError(f"ERROR: Model {self.name} not supported")

            hist_rewards.append(reward)
            pred_value.append(value)

            # Render the env
            self.model.env.render()

            # Wait a bit before the next frame unless you want to see a crazy fast video
            time.sleep(0.001)
            state = torch.FloatTensor(next_state)

        actual_return = discount_rewards(hist_rewards, self.model.gamma)

        # close the render
        self.model.env.close()
        return {
                "total_reward" : np.sum(hist_rewards),
                "total_steps"  : total_steps,
                'actual_return': actual_return,
                'pred_value'   : pred_value
        }

    def save_model(self, custom_path=None):
        if self.model is None:
            raise ValueError("ERROR: Model hasn't been created/loaded")
        if custom_path is None:
            if self.model_save_dir is None or self.model_hash is None:
                raise ValueError("ERROR: model_save_dir is None")
            os.makedirs(self.model_save_dir, exist_ok=True)
            save_path = os.path.join(self.model_save_dir, f"{self.model_hash}.pt")
        else:
            path_base = os.path.basename(custom_path)
            os.makedirs(path_base, exist_ok=True)
            save_path = custom_path
        print(f"Saving model checkpoint to: {save_path}")
        torch.save(self.model.state_dict(), save_path)

    def create_wandb_logger(self, wandb_project=None):
        wandb_project = wandb_project.strip()
        if self.model_hash is None or self.name is None:
            raise ValueError("Model hasn't been created/loaded")
        self.loggers.append(WandbLogger(project=wandb_project, name=self.model_hash,
                                        group=self.name, tags=["train"]))

    def create_csv_logger(self, log_dir):
        log_dir = log_dir.strip()
        if self.model_hash is None:
            raise ValueError("Model hasn't been created/loaded")
        self.loggers.append(CSVLogger(save_dir=log_dir, name=self.model_hash))
