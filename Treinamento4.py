import os
import wandb
import time
import random
import numpy as np
#import gym

#from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
#from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from Stablebaselines3.dummy_vec_env import DummyVecEnv

from Ambiente_SOMN.make_env import make_env
from Stablebaselines3.PPO import PPO
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
objetivo = ["Lucro", "Variabilidade", "Sustentabiliade"]
for atraso in range(-1,0,10):  ### ACMO USAR UMA COMBINAÇÃO QUE DESABILITE
#    atraso = None
    config_PPO = {
        'objetivo': 0, # 0: lucro, 1: variabilidade, 2: sustentabilidade
        'atraso': atraso,
        'batch_size': 64,
        'ent_coef': 0.00764466927690981,
        'gae_lambda': 0.9266039955653994,
        'gamma': 0.9681333537278608,
        'learning_rate': 0.07496353928768018,
        'n_epochs': 12,
        'n_steps': 3840,
        'target_kl': 0.022406375986270936
    }

    for x in range(1):    #### ACMO NUMEROS DE EXECUÇÕES COMPETIDORAS
        run1 = wandb.init(project='Fred_test_SU', #NOME DO PROJETO
                          config=config_PPO,
                          group=f"priorizando {objetivo[config_PPO['objetivo']]}", #GRUPOS A SEREM ADCIONADOS NO WANDB
                          name=f"PPO (teste 7, atraso = {atraso:02d}, run: {x + 1:02d})",
                        #   name="run_test_SU", #NOME DA EXECUÇÃO
                          save_code=True,
                          reinit=True
        )


        config = wandb.config
        env1 = DummyVecEnv([lambda: make_env(config.atraso, config.objetivo)])

        model = PPO(
            policy="MultiInputPolicy",
            env=env1,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            #clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            #vf_coef=config.vf_coef,
            #max_grad_norm=config.max_grad_norm,
            target_kl=config.target_kl,
            #stats_window_size=config.stats_window_size,
            verbose=0,
            #seed = 2023,
            device='cpu',
            #tensorboard_log=f"/content/drive/MyDrive/SOMN2/runs/{run1.id}"
            tensorboard_log=f"runs/{run1.id}"
        )
        

        model.learn(total_timesteps=166_667)
        
        # 1000 e verificar o tempo
        model.save(os.path.join(wandb.run.dir, f"model_custom_PPO (atraso = {atraso:02d}) run_{x+1:02d}"))
        wandb.finish()