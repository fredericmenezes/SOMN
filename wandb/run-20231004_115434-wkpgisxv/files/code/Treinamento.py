import os
import wandb
import time
#import gym

#from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

from Ambiente_SOMN.custom_make_env import custom_make_env
from Stable_baselines3.PPO import PPO

# Initialize a new wandb run
if len(wandb.patched["tensorboard"]) > 0:
    wandb.tensorboard.unpatch()
#wandb.tensorboard.patch(root_logdir="/content/drive/MyDrive/SOMN2/runs")
wandb.tensorboard.patch(root_logdir="./runs")


#atraso:int=None

for atraso in range(-1,0,10):  ### ACMO USAR UMA COMBINAÇÃO QUE DESABILITE
#    atraso = None
    config_PPO = {
        'atraso': atraso,
        'batch_size': 256,
        'ent_coef': 0.001641577520175419,
        'gae_lambda': 0.9142950466044,
        'gamma': 0.918623650457886,
        'learning_rate': 0.0003660144793262825,
        'n_epochs': 21,
        'n_steps': 3328,
        'target_kl': 0.02113910446426361
    }

    for x in range(1):    #### ACMO NUMEROS DE EXECUÇÕES COMPETIDORAS
        run1 = wandb.init(project='YARD', #NOME DO PROJETO
                          config=config_PPO,
                          group=f'YARDTESTES{atraso:02d}', #GRUPOS A SEREM ADCIONADOS NO WANDB
#                          name=f'custom-PPO-atraso_{atraso:02d}-run_{x+1:02d}',
                          name=f'YARD = 2', #NOME DA EXECUÇÃO
                          save_code=True,
                          reinit=True
        )

        config = wandb.config
        env1 = DummyVecEnv([lambda: custom_make_env(atraso)])

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

        model.learn(total_timesteps=3328*400)
        
        # 1000 e verificar o tempo
        model.save(os.path.join(wandb.run.dir, f"model_custom_PPO_atraso_{atraso:02d}"))
        wandb.finish()