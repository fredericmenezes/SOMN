import wandb
# import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean

# TODO
# Falta fazer essas funções capturarem as variaveis
# corretamente. (INCOMPLETO AINDA)
#  
class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        self.contador = 1
        self.ep_info_buffer = []
        self.rewards = []
        self.rw = []
        self.rw_pr = []
        self.rw_va = []
        self.rw_su = []
        self.VA = []
        self.SU = []
        self.F = []
        super(WandbCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals['infos']

        if len(infos) > 0:
            self.rewards.append(float(self.locals['rewards']))
            self.rw.append(float(infos["rw"]))
            self.rw_pr.append(float(infos["rw_pr"]))
            self.rw_va.append(float(infos["rw_va"]))
            self.rw_su.append(float(infos["rw_su"]))
            self.VA += infos["VA"]
            self.SU += infos["SU"]
            self.F += infos["F"]

        return True
    
    def _on_training_end(self) -> None:
        # Log custom metrics from the environment
        # episode_rewards = self.locals['rewards']
        infos = self.locals['infos']
        num_timesteps = self.locals['num_timesteps']

        if len(infos) > 0:
            # Accumulate custom metrics
            mean_reward_test = safe_mean([ep_info["r"] for ep_info in infos])
            ep_len_mean = safe_mean([ep_info["l"] for ep_info in infos])
            Lucro = safe_mean([ep_info["rw_pr"] for ep_info in infos])
            Variabilidade = safe_mean([ep_info["rw_va"] for ep_info in infos])
            Sutentabilidade = safe_mean([ep_info["rw_su"] for ep_info in infos])
            VA = safe_mean([va for ep_info in infos for va in ep_info["VA"]])
            SU = safe_mean([su for ep_info in infos for su in ep_info["SU"]])
            numero_de_Features = safe_mean([f for ep_info in infos for f in ep_info["F"]])
            
            # Log metrics to wandb
            wandb.log({"mean_reward_test": mean_reward_test, 
                       'timesteps': self.locals['self'].num_timesteps}
            )
            wandb.log({"ep_len_mean": ep_len_mean, 
                       'timesteps': self.locals['self'].num_timesteps}
            )
            wandb.log({"Lucro": Lucro, 
                       "timesteps": self.locals['self'].num_timesteps}
            )
            wandb.log({"Variabilidade": Variabilidade,
                       "timesteps": self.locals['self'].num_timesteps}
            )
            wandb.log({"Sutentabilidade": Sutentabilidade, 
                       "timesteps": self.locals['self'].num_timesteps}
            )
            wandb.log({"VA": VA, 
                       "timesteps": self.locals['self'].num_timesteps}
            )
            wandb.log({"SU": SU, 
                       "timesteps": self.locals['self'].num_timesteps}
            )
            wandb.log({"numero_de_Features": numero_de_Features, 
                       'timesteps': self.locals['self'].num_timesteps}
            )
            
        
        return super()._on_training_end()
