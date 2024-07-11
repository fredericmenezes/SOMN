from typing import Any, ClassVar, Dict, Optional, Tuple, Type, TypeVar, Union
import os
import yaml
import wandb
import random
from gymnasium import spaces 

import numpy as np
from torch import nn as nn
import torch

from Stablebaselines3.monitor import Monitor
from Stablebaselines3.dummy_vec_env import DummyVecEnv

from Stablebaselines3.PPO import PPO
from Stablebaselines3.DQN import DQN
from Sb3_contrib.ppo_recurrent.ppo_recurrent import RecurrentPPO
from Stablebaselines3.OffPolicyAlgorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from Ambiente_SOMN.make_env import make_env
# from stable_baselines3.common.evaluation import evaluate_policy
from Stablebaselines3.evalue_policy import evaluate_policy
from Ambiente_SOMN.Yard import Yard




class RandomPolicy(BasePolicy):
    def __init__(self, observation_space, action_space, **kwargs):
        super(RandomPolicy, self).__init__(observation_space, action_space, **kwargs)

    def _build(self):
        pass

    def forward(self, *args, **kwargs):
        pass

    def _predict(self, observation, deterministic=False):
        # observation, vectorized_env = self.obs_to_tensor(observation)

        if isinstance(observation, dict):
            n_envs = observation[next(iter(observation.keys()))].shape[0]
        else:
            n_envs = observation.shape[0]

        return torch.from_numpy(np.array([self.action_space.sample() for _ in range(n_envs)]))

class RAND(OffPolicyAlgorithm):
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "RandomPolicy": RandomPolicy,
    }
    def __init__(

        self, 
        policy, 
        env, 
        buffer_size: int = 5_000,
        batch_size: int = 16, 
        gradient_steps: int = 1,
        **kwargs
    ):
        super(RAND, self).__init__(
            policy, 
            env, 
            buffer_size, 
            batch_size, 
            gradient_steps, 
            **kwargs
        )
        self._setup_model()


    def train(self, gradient_steps, batch_size):
        pass

    def _setup_model(self):
        self.set_random_seed(self.seed)
        self.policy = RandomPolicy(self.observation_space, self.action_space)

        if self.replay_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.replay_buffer_class = DictReplayBuffer
            else:
                self.replay_buffer_class = ReplayBuffer
        

        if self.replay_buffer is None:
            # Make a local copy as we should not pickle
            # the environment when using HerReplayBuffer
            replay_buffer_kwargs = self.replay_buffer_kwargs.copy()
            if issubclass(self.replay_buffer_class, HerReplayBuffer):
                assert self.env is not None, "You must pass an environment when using `HerReplayBuffer`"
                replay_buffer_kwargs["env"] = self.env
            self.replay_buffer = self.replay_buffer_class(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                device=self.device,
                n_envs=self.n_envs,
                optimize_memory_usage=self.optimize_memory_usage,
                **replay_buffer_kwargs,  # pytype:disable=wrong-keyword-args
            )

        # self.replay_buffer = ReplayBuffer(
        #     self.buffer_size,
        #     self.observation_space,
        #     self.action_space,
        #     device=self.device,
        #     n_envs=self.n_envs,
        # )

    def learn(self, total_timesteps, callback=None, log_interval=4, tb_log_name="RAND", reset_num_timesteps=True):
        self._setup_learn(total_timesteps, callback, reset_num_timesteps, tb_log_name)

        while self.num_timesteps < total_timesteps:
            action = np.array([self.action_space.sample() * 0 for _ in range(self.env.num_envs)])
            new_obs, rewards, dones, infos = self.env.step(action)
            self.num_timesteps += self.env.num_envs
            self.replay_buffer.add(self._last_obs, new_obs, action, rewards, dones, infos)
            self._last_obs = new_obs

            self._update_info_buffer(infos, dones)

            if dones.any():
                self._dump_logs()
                acoes = action.tolist()
                wandb.log({'Actions':  np.mean(acoes),
                        'timesteps': self.num_timesteps,
                        'mean_reward_test': safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]),
                        }
                )
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
    elif alg_name == 'randomico':
        model = alg_class(
            policy="RandomPolicy",
            env=env,
            buffer_size=config.buffer_size,
            verbose=0,
            device='cpu'
        )
        return model    


def train_and_select_best(experimento, alg_class, alg_name, config, n_evaluations, total_timesteps):
    best_mean_reward = -float('inf')
    best_model = None

    
    for i in range(n_evaluations):
        wandb_config = config

        run = wandb.init(
             project=wandb_config['projeto'],
             config = wandb_config,
             group = wandb_config['grupo'],
             name = f"{alg_name}_{experimento}_run_{i + 1:02d}",
             save_code = True,
             reinit = True
        )
        wandb_config = wandb.config
        print(wandb_config)
        env = DummyVecEnv([lambda: make_env(wandb_config.atraso, wandb_config.objetivo)])

        print(f"Training {alg_name} model {i+1}/{n_evaluations}")
        model = config_alg_parameters(env, alg_class, alg_name, wandb_config, run)
        model.learn(total_timesteps=total_timesteps)

        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
        print(f"Mean reward for {alg_name} model {i+1}/{n_evaluations}: {mean_reward} ± {std_reward}")

        
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            best_model = model
            num = i + 1

        model.save(os.path.join("wandb_models", f"{alg_name}_{experimento}_run_{i + 1:02d}"))
        wandb.finish()

    best_model.save(os.path.join("best_model", f"{alg_name}_{experimento}_run_{num:02d}"))

    return best_mean_reward, num


if __name__ == "__main__":
    
    n_evaluations = 1
    total_timesteps = 1_000_000

    # Initialize a new wandb run
    if len(wandb.patched["tensorboard"]) > 0:
        wandb.tensorboard.unpatch()

    wandb.tensorboard.patch(root_logdir="./runs")

    # Set up your default hyperparameters
    with open("./config_result.yaml") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

    # best_ppo_lstm, best_mean_reward, num_best_ppo_lstm = train_and_select_best(RecurrentPPO, 
    #                                                          "ppo_lstm", 
    #                                                          config["ppo_lstm"], 
    #                                                          n_evaluations, 
    #                                                          total_timesteps)
    # best_dqn, best_mean_reward, num_best_dqn = train_and_select_best(DQN, 
    #                                                "dqn", 
    #                                                config["dqn"], 
    #                                                n_evaluations, 
    #                                                total_timesteps)
    # best_ppo, best_mean_reward, num_best_ppo = train_and_select_best(PPO, 
    #                                                "ppo", 
    #                                                config["ppo"], 
    #                                                n_evaluations, 
    #                                                total_timesteps)
    random_mean_reward, num_best_random = train_and_select_best("comp10_act=0_t=300",
                                                                RAND,  
                                                                "randomico", 
                                                                config["randomico"], 
                                                                n_evaluations, 
                                                                total_timesteps)
    
    # print(f" Os melhores modelos são: PPO Recorrente-{num_best_ppo_lstm}, DQN-{num_best_dqn}, PPO-{num_best_ppo}")
    


