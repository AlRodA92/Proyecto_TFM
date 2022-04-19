from pygame import K_0
from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import sys

# Get the multitask functions
sys.path.insert(0,'/home/aroman/TFM/Código/Proyecto_TFM/Proyecto_TFM/src/Multi_task')

from rewards_multitask import reward_task_objetive, reward_task_survive, reward_task_kill, reward_scalarization

class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated_total = False
        terminated = False
        terminated_task = False
        episode_return = 0
        task_reward = 0
        extrinsic_reward = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated_total:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            reward, terminated, env_info = self.env.step(actions[0])

            # Check if the multi-objetive option is set and compute the aditional reward and
            # the total reward scalarization
            if self.args.task == 'objetive':
                if not terminated_task:
                    reward_objetive,terminated_task = reward_task_objetive(self,terminated)
                if all([terminated,terminated_task]):
                    terminated_total = True
                reward, reward_ex, reward_objetive = reward_scalarization(self,reward,reward_objetive)
                task_reward += reward_objetive
                extrinsic_reward += reward_ex
            elif self.args.task == 'kill':
                reward_kill = reward_task_kill(self)
                reward, reward_ex, reward_kill = reward_scalarization(self,reward,reward_kill)
                task_reward += reward_kill
                extrinsic_reward += reward_ex
                
            episode_return += reward
    
            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated_total != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        # Check survive task is selected
        if self.args.task == "survive":
            reward_survive = reward_task_survive(self)
            episode_return, reward_ex, reward_survive = reward_scalarization(self,reward,reward_survive)
            task_reward += reward_survive
            extrinsic_reward += reward_ex

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        # Add task and extrinsic rewards to metrics
        if self.args.task != '':
            cur_stats['reward_ext'] = extrinsic_reward
            cur_stats['reward_' + self.args.task] = task_reward

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                if 'reward' in k:
                    self.logger.log_stat(prefix + k + "_mean", np.mean(v), self.t_env)
                    self.logger.log_stat(prefix + k + "_std", np.std(v), self.t_env)
                else:
                    self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()