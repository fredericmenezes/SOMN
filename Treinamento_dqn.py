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
from Stablebaselines3.PPO import PPO
from Stablebaselines3.DQN import DQN
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
    config_DQN = {
        'objetivo': 0, # 0: lucro, 1: variabilidade, 2: sustentabilidade
        'experimento': experimento,
        'atraso': atraso,
        'batch_size': 256,
        'gamma': 0.91,
        'learning_rate': 0.0017,
        'buffer_size': 50000,
        'learning_starts': 1000,
        'target_update_interval': 10000,
        'train_freq': 4,
        'gradient_steps': 1,
        'exploration_fraction': 0.1,
        'exploration_initial_eps': 1.0,
        'exploration_final_eps': 0.02,
        'max_grad_norm': 10
    }
    # classstable_baselines3.dqn.DQN(policy, 
    # env, 
    # learning_rate=0.0001, 
    # buffer_size=1000000, 
    # learning_starts=100, 
    # batch_size=32, 
    # tau=1.0, 
    # gamma=0.99, 
    # train_freq=4, 
    # gradient_steps=1, 
    # replay_buffer_class=None, 
    # replay_buffer_kwargs=None, 
    # optimize_memory_usage=False, 
    # target_update_interval=10000, 
    # exploration_fraction=0.1, 
    # exploration_initial_eps=1.0, 
    # exploration_final_eps=0.05, 
    # max_grad_norm=10, 
    # stats_window_size=100, 
    # tensorboard_log=None, 
    # policy_kwargs=None, 
    # verbose=0, 
    # seed=None, 
    # device='auto', 
    # _init_setup_model=True)


    for x in range(5):    #### ACMO NUMEROS DE EXECUÇÕES COMPETIDORAS
        run1 = wandb.init(
            project='Fred_test_SU', #NOME DO PROJETO
            config=config_DQN,
            group=f"priorizando {objetivo[config_DQN['objetivo']]}", #GRUPOS A SEREM ADCIONADOS NO WANDB
        #   name=f"PPO (teste 12, atraso = {atraso:02d}, run: {x + 1:02d})",
            name=f"DQN (teste 02, experimento {experimento:02d}, run {x + 1:02d})",
        #   name="run_test_SU", #NOME DA EXECUÇÃO
            save_code=True,
            reinit=True
        )


        config = wandb.config
        env1 = DummyVecEnv([lambda: make_env(config.atraso, config.objetivo)])

        model = DQN(
            policy="MultiInputPolicy",
            env=env1,
            batch_size=config.batch_size,
            gamma=config.gamma,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            learning_starts=config.learning_starts,
            target_update_interval=config.target_update_interval,
            train_freq=config.train_freq,
            gradient_steps=config.gradient_steps,
            exploration_fraction=config.exploration_fraction,
            # exploration_initial_eps=config.exploration_initial_eps,
            exploration_final_eps=config.exploration_final_eps,
            max_grad_norm=config.max_grad_norm,
            verbose=0,
            device='cpu',
            tensorboard_log=f"runs/{run1.id}"
        )
        

        model.learn(total_timesteps=3_000_000)
        
        # 1000 e verificar o tempo
        # model.save(os.path.join(wandb.run.dir, f"model_custom_PPO (atraso = {atraso:02d}) run_{x+1:02d}"))
        model.save(os.path.join(wandb.run.dir, f"Sweep_DQN_exp_{experimento:02d}_param_0 (atraso = {atraso:02d}) run_{x + 1:02d}"))
        wandb.finish()