import wandb
import yaml

import os
import time
import random
import numpy as np
#import gym
from torch import nn as nn

#from stable_baselines3 import PPO
from Stablebaselines3.monitor import Monitor
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from Stablebaselines3.dummy_vec_env import DummyVecEnv

from Ambiente_SOMN.make_env import make_env
# from Stablebaselines3.PPO import PPO
from Sb3_contrib.ppo_recurrent.ppo_recurrent import RecurrentPPO
from Ambiente_SOMN.Yard import Yard

# Initialize a new wandb run
if len(wandb.patched["tensorboard"]) > 0:
    wandb.tensorboard.unpatch()
#wandb.tensorboard.patch(root_logdir="/content/drive/MyDrive/SOMN2/runs")
wandb.tensorboard.patch(root_logdir="./runs")

# seed
#random.seed(10)
#np.random.seed(1)
from Seed.Seed import seed_everything

#seed_everything(2023)

#atraso:int=None
# objetivo = ["Lucro", "Variabilidade", "Sustentabiliade"]
experimento = 0
for atraso in range(-1,0,10):  ### ACMO USAR UMA COMBINAÇÃO QUE DESABILITE
    experimento += 1
#    atraso = None
    # config_PPO = {
    #     'objetivo': 2, # 0: lucro, 1: variabilidade, 2: sustentabilidade
    #     'atraso': atraso,
    #     'batch_size': 256,
    #     'ent_coef': 0.001641577520175419,
    #     'gae_lambda': 0.9142950466044,
    #     'gamma': 0.918623650457886,
    #     'learning_rate': 0.0003660144793262825,
    #     'n_epochs': 21,
    #     'n_steps': 3328,
    #     'target_kl': 0.02113910446426361
    # }

    # Set up your default hyperparameters
    with open("./config_ppo_lstm.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    for x in range(1):    #### ACMO NUMEROS DE EXECUÇÕES COMPETIDORAS

        

        run1 = wandb.init(project="Sweep_PPO_Recurrent", #NOME DO PROJETO
                          config=config,
                          group="PPO_Recurrent", #GRUPOS A SEREM ADCIONADOS NO WANDB
#                          name=f'custom-PPO-atraso_{atraso:02d}-run_{x+1:02d}',
                        #   name=f"PPO (sintonia 1)",
                          save_code=True,
                          reinit=True
        )


        config = wandb.config
        env1 = DummyVecEnv([lambda: make_env(config.atraso, config.objetivo)])

        model = RecurrentPPO(
            policy="MultiInputLstmPolicy",
            env=env1,
            batch_size=config.batch_size,
            n_steps=config.n_steps,
            gamma=config.gamma,
            learning_rate=config.learning_rate,
            ent_coef=config.ent_coef,
            clip_range=config.clip_range,
            n_epochs=config.n_epochs,
            gae_lambda=config.gae_lambda,
            max_grad_norm=config.max_grad_norm,
            vf_coef=config.vf_coef,
            # clip_range_vf=config.clip_range_vf,
            target_kl=config.target_kl,
            #stats_window_size=config.stats_window_size,
            verbose=0,
            #seed = 2023,
            device='cpu',
            #tensorboard_log=f"/content/drive/MyDrive/SOMN2/runs/{run1.id}"
            tensorboard_log=f"runs/{run1.id}"
        )
        
        multiplicador_passos = int(1_000_000) / config.n_steps
        # model.learn(total_timesteps=3328*301)
        model.learn(total_timesteps = config.n_steps * (multiplicador_passos + 1))
        
        # 1000 e verificar o tempo
        model.save(os.path.join(wandb.run.dir, "model_PPO_lstm_lucro"))
        wandb.finish()