import wandb
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces

# from stable_baselines3.common.base_class import BaseAlgorithm
from Stablebaselines3.base_class import BaseAlgorithm

from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv

SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")


class OnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    rollout_buffer: RolloutBuffer
    policy: ActorCriticPolicy

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "cuda",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            seed=seed,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )
        #self.acoes
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, spaces.Dict) else RolloutBuffer

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        # pytype:disable=not-instantiable
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        )
        # pytype:enable=not-instantiable
        self.policy = self.policy.to(self.device)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs
            

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
        self: SelfOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOnPolicyAlgorithm:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                    # --------- WandB Log ----------- #
                    # para lembrar como estão as variaveis em info:
                    # info = {"rw": reward,
                    #         "rw_pr": rw_pr,                   # reward lucro
                    #         "rw_va": rw_va,                   # reward variabilidade
                    #         "rw_su": rw_su,                   # reward sustentabilidade
                    #         "VA": variabilidade,
                    #         "SU": sustentabilidade,
                    #         "F": F,                           # numero de features (maquinas)
                    #         "acoes": acoes,
                    #         "atrasos_reais": atrasos_reais,   # atrasos para comparar com acoes
                    #         "acao_on_state_plan": self.acao_on_state_plan,
                    #         "carga_on_state_plan": self.carga_on_state_plan,
                    #         "patio_on_state_plan": self.patio_on_state_plan
                    #        }
                    wandb.log({"mean_reward_test": safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]),'timesteps': self.num_timesteps})
                    wandb.log({"ep_len_mean": safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]),'timesteps': self.num_timesteps})
                    #wandb.log({"recompensa": safe_mean([ep_info["rw"] for ep_info in self.ep_info_buffer]),'timesteps': self.num_timesteps})
                    # wandb.log({"recompensa_lucro": safe_mean([ep_info["rw_pr"] for ep_info in self.ep_info_buffer]),
                    #            "recompensa_variabilidade": safe_mean([ep_info["rw_va"] for ep_info in self.ep_info_buffer]),
                    #            "recompensa_sustentabilidade": safe_mean([ep_info["rw_su"] for ep_info in self.ep_info_buffer]),
                    #            'timesteps': self.num_timesteps})
                    wandb.log({"Lucro":safe_mean([ep_info["rw_pr"] for ep_info in self.ep_info_buffer]),"timesteps": self.num_timesteps})
                    wandb.log({"Variabilidade":safe_mean([ep_info["rw_va"] for ep_info in self.ep_info_buffer]),"timesteps": self.num_timesteps})
                    wandb.log({"Sutentabilidade":safe_mean([ep_info["rw_su"] for ep_info in self.ep_info_buffer]),"timesteps": self.num_timesteps})
                    # wandb.log({"variabilidade": safe_mean([ep_info["VA"] for ep_info in self.ep_info_buffer]),
                    #            "sustentabilidade": safe_mean([ep_info["SU"] for ep_info in self.ep_info_buffer]),
                    #            'timesteps': self.num_timesteps})
                    wandb.log({"VA": safe_mean([va for ep_info in self.ep_info_buffer for va in ep_info["VA"]]), "timesteps": self.num_timesteps})
                    wandb.log({"SU": safe_mean([su for ep_info in self.ep_info_buffer for su in ep_info["SU"]]), "timesteps": self.num_timesteps})
                    # if self.num_timesteps == 1000000:
                    #     F = []
                    #     F = [num_features for ep_info in self.ep_info_buffer for num_features in ep_info["F"]]
                    #     hist_F = np.histogram(F)
                    #     wandb.log({"numero_de_Features": wandb.Histogram(np_histogram=hist_F, num_bins=10),'timesteps': self.num_timesteps})
                    wandb.log({"numero_de_Features": safe_mean([f for ep_info in self.ep_info_buffer for f in ep_info["F"]]), 'timesteps': self.num_timesteps})
                    # cargas_on_state_plan = []
                    # cargas_on_state_plan = [carga_on_state_plan for ep_info in self.ep_info_buffer for carga_on_state_plan in ep_info["carga_on_state_plan"]]
                    # for carga in cargas_on_state_plan:
                    #     wandb.log({"carga_on_state_plan": carga,"timesteps": self.num_timesteps})
                    
                    # patios_on_state_plan = []
                    # patios_on_state_plan = [patio_on_state_plan for ep_info in self.ep_info_buffer for patio_on_state_plan in ep_info["patio_on_state_plan"]]
                    # for patio in patios_on_state_plan:
                    #     wandb.log({"patio_on_state_plan": patio,"timesteps": self.num_timesteps})

                    if self.num_timesteps > 1 and self.contador == 0 or\
                       self.num_timesteps > 9000 and self.contador == 1 or\
                       self.num_timesteps > 49000 and self.contador == 2 or\
                       self.num_timesteps > 99000 and self.contador == 3 or\
                       self.num_timesteps > 149000 and self.contador == 4 or\
                       self.num_timesteps > 499000 and self.contador == 5 or\
                       self.num_timesteps > 990000 and self.contador == 6:
                    # if self.num_timesteps > 1 and self.num_timesteps < 5000 or\
                    #    self.num_timesteps > 9000 and self.num_timesteps < 10000 or\
                    #    self.num_timesteps > 49000 and self.num_timesteps < 50000 or\
                    #    self.num_timesteps > 99000 and self.num_timesteps < 100000 or\
                    #    self.num_timesteps > 249000 and self.num_timesteps < 250000 or\
                    #    self.num_timesteps > 499000 and self.num_timesteps < 500000 or\
                    #    self.num_timesteps > 990000 and self.num_timesteps < 1000000:
                        self.contador += 1
                        acoes = []
                        acoes_on_state_plan = []
                        atrasos = []
                        acoes = [acao for ep_info in self.ep_info_buffer for acao in ep_info["acoes"]]
                        acoes_on_state_plan = [acao_on_state_plan for ep_info in self.ep_info_buffer for acao_on_state_plan in ep_info["acao_on_state_plan"]]
                        atrasos = [atraso for ep_info in self.ep_info_buffer for atraso in ep_info["atrasos_reais"]]
                        # hist_acoes = np.histogram(acoes)
                        # hist_acoes_on_state_plan = np.histogram(acoes_on_state_plan)
                        # hist_atrasos = np.histogram(atrasos)
                        # wandb.log({f"acoes (timesteps = {self.num_timesteps})": wandb.Histogram(np_histogram=hist_acoes, num_bins=100),
                        #            f"acoes_on_state_plan (timesteps = {self.num_timesteps})": wandb.Histogram(np_histogram=hist_acoes_on_state_plan, num_bins=100),
                        #            f"atrasos (timesteps = {self.num_timesteps})": wandb.Histogram(np_histogram=hist_atrasos, num_bins=100),
                        #             'timesteps': self.num_timesteps}
                        # )
                        # exemplo
                        # data = [[s] for s in bird_scores]
                        # table = wandb.Table(data=data, columns=["bird_scores"])
                        # wandb.log({'my_histogram': wandb.plot.histogram(table, "bird_scores",
                        #         title="Bird Confidence Scores")})
                        data_acoes = [[i, acoes[i]] for i in range(len(acoes))]
                        table_acoes = wandb.Table(data=data_acoes, columns=["step", "acoes"])
                        fields_acoes = {"value" : "acoes",  "title" : "Ações timesteps = " + str(self.num_timesteps)}
                        custom_acoes_histogram = wandb.plot_table(
                            vega_spec_name="lacmor/histograma_preset_9",
                            data_table = table_acoes,
                            fields = fields_acoes)
                        wandb.log({"acoes (timesteps = " + str(self.num_timesteps) + ")": custom_acoes_histogram})

                        # wandb.log({
                        #             "acoes timesteps = " + self.num_timesteps: wandb.plot.histogram(table_acoes, "acoes",
                        #             title="Ações timesteps = " + self.num_timesteps)
                        #             }
                        # )
                        
                        data_atrasos = [[i, acoes[i], atrasos[i]] for i in range(len(atrasos))]
                        table_atrasos = wandb.Table(data=data_atrasos, columns=["step", "acoes", "atrasos"])
                        fields_atrasos = {"value" : "atrasos",  "title" : "Atrasos reais timesteps = " + str(self.num_timesteps)}
                        custom_atrasos_histogram1 = wandb.plot_table(
                            vega_spec_name="lacmor/histograma_preset_9",
                            data_table = table_atrasos,
                            fields = fields_atrasos)
                        custom_atrasos_histogram2 = wandb.plot_table(
                            vega_spec_name="lacmor/histograma_preset_9",
                            data_table = table_atrasos,
                            fields = fields_acoes)
                        wandb.log({"atrasos (timesteps = " + str(self.num_timesteps) + ")": custom_atrasos_histogram1, "acoes (timesteps = " + str(self.num_timesteps) + ")": custom_atrasos_histogram2})
                        # wandb.log({
                        #             "atrasos reais timesteps = " + self.num_timesteps: wandb.plot.histogram(table_atrasos, "atrasos",
                        #             title="Atrasos reais timesteps = " + self.num_timesteps)
                        #             #f"acoes (timesteps = {self.num_timesteps})": wandb.plot.histogram(table, "acoes",
                        #             #title=f"Ações (timesteps = {self.num_timesteps})")
                        #          }
                        # )

                        #wandb.log({f"atrasos (timesteps = {self.num_timesteps})": wandb.Histogram(np_histogram=hist_atrasos, num_bins=100),'timesteps': self.num_timesteps})
                        
                        
                    
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)
                # --------- WandB Log ----------- #
                #wandb.log({"total_timesteps": self.num_timesteps})

                # --------- WandB Log ----------- #
                #wandb.log({'reward': safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]), 'timesteps': self.num_timesteps})

                # Set up data to log in custom charts
                #global acoes
                #todas_acoes.append([iteration, acoes])



            #del globals()['acoes']
            self.train()


        # Create a table with the columns to plot
        #table = wandb.Table(data=todas_acoes, columns=["step", "acao"])

        # Use the table to populate various custom charts
        #line_plot = wandb.plot.line(table, x='step', y='height', title='Line Plot')
        #histogram = wandb.plot.histogram(table, value='height', title='Histogram')
        #scatter = wandb.plot.scatter(table, x='step', y='acao', title='Acões escolhidas')

        # Log custom tables, which will show up in customizable charts in the UI
        #wandb.log({#'line_1': line_plot,
                    #'histogram_1': histogram,
                    #'scatter_1': scatter})

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []