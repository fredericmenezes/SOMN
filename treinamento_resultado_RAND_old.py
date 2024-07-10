import os
import yaml
import wandb
import random

import numpy as np
from torch import nn as nn

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from stable_baselines3 import PPO, DQN
from sb3_contrib.ppo_recurrent import RecurrentPPO

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import safe_mean

from Ambiente_SOMN.make_env import make_env
from stable_baselines3.common.evaluation import evaluate_policy
from Ambiente_SOMN.Yard import Yard


class RandomPolicy(BasePolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super(RandomPolicy, self).__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule):
        pass

    def forward(self, *args, **kwargs):
        pass

    def _predict(self, observation, deterministic=False):
        return np.array([self.action_space.sample() for _ in range(observation.shape[0])])


class RAND(OffPolicyAlgorithm):
    def __init__(self, policy, env, **kwargs):
        super(RAND, self).__init__(policy, env, **kwargs)

    def train(self, gradient_steps, batch_size):
        pass

    def _setup_model(self):
        self.set_random_seed(self.seed)
        self.policy = RandomPolicy(self.observation_space, self.action_space, self.lr_schedule)
        self.replay_buffer = ReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            device=self.device,
            n_envs=self.n_envs,
        )

    def learn(self, total_timesteps, callback=None, log_interval=4, tb_log_name="RAND", reset_num_timesteps=True):
        self._setup_learn(total_timesteps, callback, reset_num_timesteps, tb_log_name)
        while self.num_timesteps < total_timesteps:
            action = np.array([self.action_space.sample() for _ in range(self.env.num_envs)])
            new_obs, rewards, dones, infos = self.env.step(action)
            self.num_timesteps += self.env.num_envs
            self.replay_buffer.add(self._last_obs, new_obs, action, rewards, dones)
            self._last_obs = new_obs
            if dones.any():
                self._last_obs = self.env.reset()
        return self


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
            verbose=0,
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
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            target_kl=config.target_kl,
            verbose=0,
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
            exploration_final_eps=config.exploration_final_eps,
            max_grad_norm=config.max_grad_norm,
            verbose=0,
            device='cpu',
            tensorboard_log=f"runs/{run.id}"
        )
        return model
    elif alg_name == 'randomico':
        model = alg_class(
            policy="RandomPolicy",
            env=env,
            buffer_size=config.buffer_size,
            verbose=0,
            device='cpu'
        )
        return model


def train_and_select_best(alg_class, alg_name, config, n_evaluations, total_timesteps):
    best_mean_reward = -float('inf')
    best_model = None

    for i in range(n_evaluations):
        wandb_config = config
        run = wandb.init(
            project=wandb_config['projeto'],
            config=wandb_config,
            group=wandb_config['grupo'],
            name=f"{alg_name}_comp4_run_{i + 1:02d}",
            save_code=True,
            reinit=True
        )
        wandb_config = wandb.config
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

        model.save(os.path.join("wandb_models", f"{alg_name}_comp4_run_{i + 1:02d}"))
        wandb.finish()

    best_model.save(os.path.join("best_model", f"{alg_name}_comp4_run_{num:02d}"))
    return best_model, best_mean_reward, num


if __name__ == "__main__":
    n_evaluations = 5
    total_timesteps = 1_000_000

    if len(wandb.patched["tensorboard"]) > 0:
        wandb.tensorboard.unpatch()

    wandb.tensorboard.patch(root_logdir="./runs")

    with open("./config_result.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    best_ppo_lstm, best_mean_reward, num_best_ppo_lstm = train_and_select_best(RecurrentPPO, 
                                                                              "ppo_lstm", 
                                                                              config["ppo_lstm"], 
                                                                              n_evaluations, 
                                                                              total_timesteps)
    best_dqn, best_mean_reward, num_best_dqn = train_and_select_best(DQN, 
                                                                     "dqn", 
                                                                     config["dqn"], 
                                                                     n_evaluations, 
                                                                     total_timesteps)
    best_ppo, best_mean_reward, num_best_ppo = train_and_select_best(PPO, 
                                                                     "ppo", 
                                                                     config["ppo"], 
                                                                     n_evaluations, 
                                                                     total_timesteps)
    random_mean_reward, num_best_random = train_and_select_best(RAND, 
                                                                "randomico", 
                                                                config["randomico"], 
                                                                n_evaluations, 
                                                                total_timesteps)

    print(f"Os melhores modelos são: PPO Recorrente-{num_best_ppo_lstm}, DQN-{num_best_dqn}, PPO-{num_best_ppo}, Randomico-{num_best_random}")
