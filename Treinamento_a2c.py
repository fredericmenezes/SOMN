import os
import wandb
import time
import random
import numpy as np
#import gym

#from stable_baselines3 import PPO
# from stable_baselines3.common.monitor import Monitor
from Stablebaselines3.monitor import Monitor
#from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from Stablebaselines3.dummy_vec_env import DummyVecEnv

from Ambiente_SOMN.make_env import make_env
from Stablebaselines3.a2c import A2C
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
experimento = 0
for atraso in range(-1,0,10):  ### ACMO USAR UMA COMBINAÇÃO QUE DESABILITE
#    atraso = None
    experimento += 1
    # Configuração do experimento
    config_A2C = {
        'objetivo': 0,  # 0: lucro, 1: variabilidade, 2: sustentabilidade
        'experimento': 1,  # Defina seu experimento específico aqui
        'atraso': 10,  # Defina o atraso específico aqui
        'learning_rate': 0.0007,
        'n_steps': 5,
        'gamma': 0.99,
        'gae_lambda': 1.0,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'use_rms_prop': True,
        'rms_prop_eps': 1e-5,
    }


    for x in range(5):    #### ACMO NUMEROS DE EXECUÇÕES COMPETIDORAS
        run1 = wandb.init(
            project='Fred_test_SU', #NOME DO PROJETO
            config=config_A2C,
            group=f"priorizando {objetivo[config_A2C['objetivo']]}", #GRUPOS A SEREM ADCIONADOS NO WANDB
        #   name=f"PPO (teste 12, atraso = {atraso:02d}, run: {x + 1:02d})",
            name=f"A2C (teste 01, experimento {experimento:02d}, run {x + 1:02d})",
        #   name="run_test_SU", #NOME DA EXECUÇÃO
            save_code=True,
            reinit=True
        )


        config = wandb.config
        env1 = DummyVecEnv([lambda: make_env(config.atraso, config.objetivo)])

        model = A2C(
            policy="MultiInputPolicy",
            env=env1,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            use_rms_prop=config.use_rms_prop,
            rms_prop_eps=config.rms_prop_eps,
            verbose=0,
            device='cpu',
            tensorboard_log=f"runs/{run1.id}"
        )
        

        model.learn(total_timesteps=3_000_000)
        
        # 1000 e verificar o tempo
        # model.save(os.path.join(wandb.run.dir, f"model_custom_PPO (atraso = {atraso:02d}) run_{x+1:02d}"))
        model.save(os.path.join(wandb.run.dir, f"Sweep_A2C_exp_{experimento:02d} (atraso = {atraso:02d}) run_{x + 1:02d}"))
        wandb.finish()