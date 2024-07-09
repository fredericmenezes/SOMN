import os
import yaml
import wandb
import random

import numpy as np
from torch import nn as nn

from Stablebaselines3.monitor import Monitor
from Stablebaselines3.dummy_vec_env import DummyVecEnv

from Stablebaselines3.PPO import PPO
from Stablebaselines3.DQN import DQN
from Sb3_contrib.ppo_recurrent.ppo_recurrent import RecurrentPPO

from Ambiente_SOMN.make_env import make_env
from stable_baselines3.common.evaluation import evaluate_policy
from Ambiente_SOMN.Yard import Yard

# def seed_everything(seed):
#     random.seed(seed)
#     np.random.seed(seed)

def config_alg_parameters(env, alg_class, alg_name, config, run):
    
    if alg_name == 'ppo_lstm':    
        model = alg_class(
            policy="MultiInputLstmPolicy",
            env=env,
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
            # target_kl=config.target_kl,
            # stats_window_size=config.stats_window_size,
            verbose=0,
            # seed = 2023,
            device='cpu',
            tensorboard_log=f"runs/{run.id}"
        )
        return model
    elif alg_name == 'ppo':

        model = alg_class(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            # clip_range_vf=config.clip_range_vf,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            target_kl=config.target_kl,
            # stats_window_size=config.stats_window_size,
            verbose=0,
            # seed = 2023,
            device='cpu',
            tensorboard_log=f"runs/{run.id}"
        )
        return model
    elif alg_name == 'dqn':
        model = alg_class(
            policy="MultiInputPolicy",
            env=env,
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
            tensorboard_log=f"runs/{run.id}"
        )
        return model


def train_and_select_best(alg_class, alg_name, config, n_evaluations, total_timesteps):
    best_mean_reward = -float('inf')
    best_model = None

    
    for i in range(n_evaluations):
        wandb_config = config

        run = wandb.init(
             project=wandb_config['projeto'],
             config = wandb_config,
             group = wandb_config['grupo'],
             name = f"{alg_name}_comp1_run_{i + 1:02d}",
             save_code = True,
             reinit = True
        )
        wandb_config = wandb.config
        print(wandb_config)
        env = DummyVecEnv([lambda: make_env(wandb_config.atraso, wandb_config.objetivo)])

        print(f"Training {alg_name} model {i+1}/{n_evaluations}")
        model = config_alg_parameters(env, alg_class, alg_name, wandb_config, run)
        model.learn(total_timesteps=total_timesteps)
        
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        print(f"Mean reward for {alg_name} model {i+1}/{n_evaluations}: {mean_reward} ± {std_reward}")

        
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            best_model = model
            num = i + 1

        model.save(os.path.join("wandb_models", f"{alg_name}_comp1_run_{i + 1:02d}"))
        wandb.finish()

    best_model.save(os.path.join("best_model", f"{alg_name}_comp1_run_{i + 1:02d}"))

    return best_model, best_mean_reward, num


if __name__ == "__main__":
    
    # seed_everything(2024)

    n_evaluations = 1
    total_timesteps = 1_000_000

    # Initialize a new wandb run
    if len(wandb.patched["tensorboard"]) > 0:
        wandb.tensorboard.unpatch()

    wandb.tensorboard.patch(root_logdir="./runs")

    # Set up your default hyperparameters
    with open("./config_result.yaml") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

    best_ppo_lstm, best_mean_reward, num_best_ppo_lstm = train_and_select_best(RecurrentPPO, 
                                                             "ppo_lstm", 
                                                             config["ppo_lstm"], 
                                                             n_evaluations, 
                                                             total_timesteps)
    # best_dqn, num_best_dqn = train_and_select_best(DQN, 
    #                                                "dqn", 
    #                                                config["dqn"], 
    #                                                n_evaluations, 
    #                                                total_timesteps)
    # best_ppo, num_best_ppo = train_and_select_best(PPO, 
    #                                                "ppo", 
    #                                                config["ppo"], 
    #                                                n_evaluations, 
    #                                                total_timesteps)
    
    # print(f" Os melhores modelos são: {num_best_ppo_lstm}, {num_best_dqn}, {num_best_ppo}")
    print(f" O  modelo gerado é: {num_best_ppo_lstm} com o melhor reward de {best_mean_reward} em média.")


