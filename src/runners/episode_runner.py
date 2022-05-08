from pygame import K_0
from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import sys

# Get the multitask functions
sys.path.insert(0,'/home/aroman/Proyecto_TFM/Proyecto_TFM/src/Multi_task')

from rewards_multitask import reward_task_objetive, reward_task_survive, reward_task_kill, reward_scalarization, check_ally_death

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

        if self.args.task != 'None':
            self.train_returns = [[] for i in range(3)]
            self.test_returns = [[] for i in range(3)]
        else:
            self.train_returns = []
            self.test_returns = []

        self.train_stats = {}
        self.test_stats = {}
        
        self.task_train_stats = {'dist':[],'objetive_reach':[]}
        self.task_test_stats = {'dist':[],'objetive_reach':[]}

        if self.args.task == 'objetive':
            self.task_train_stats = {'dist':[],'objetive_reach':[]}
            self.task_test_stats = {'dist':[],'objetive_reach':[]}
        elif self.args.task == 'kill':
            self.task_train_stats = {'number_kill':[],'kill':[]}
            self.task_test_stats = {'number_kill':[],'kill':[]}
        elif self.args.task == 'survive': 
            self.task_train_stats = {'survive':[],'number_death':[]}
            self.task_test_stats = {'survive':[],'number_death':[]}

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

        terminated = False
        episode_return = 0
        task_reward = 0
        smac_reward = 0
        task_reach = 0
        dist_prev = 0
        number_death = 0
        number_kill = 0
        survive = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

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
                time = self.t
                reward_objetive,dist,dist_prev = reward_task_objetive(self,task_reward,dist_prev,time)
                if dist < self.args.eps_objetive:
                    task_reach = 1
                    terminated = True
                else:
                    task_reach = 0
                rewards = reward_scalarization(self,reward,reward_objetive)
                episode_return += rewards[0]
                smac_reward += rewards[1]
                task_reward += rewards[2]
            elif self.args.task == 'kill':
                reward_kill, death, number_kill = reward_task_kill(self,number_kill)
                rewards = reward_scalarization(self,reward,reward_kill)
                episode_return += rewards[0]
                smac_reward += rewards[1]
                task_reward += rewards[2]
            # Check survive task is selected
            elif self.args.task == "survive":
                reward_survive, survive, number_death = reward_task_survive(self,terminated,number_death)
                rewards = reward_scalarization(self,reward,reward_survive)
                episode_return += rewards[0]
                smac_reward += rewards[1]
                task_reward += rewards[2]
                
            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

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
        task_stats = self.task_test_stats if test_mode else self.task_train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        # Add task and extrinsic rewards to metrics
        if self.args.task == 'objetive':
            task_stats['dist'].append(dist)
            task_stats['objetive_reach'].append(task_reach)
        elif self.args.task == 'kill':
            task_stats['number_kill'].append(number_kill)
            task_stats['kill'].append(death)
        elif self.args.task == 'survive': 
            task_stats['survive'].append(survive)
            task_stats['number_death'].append(number_death)
        
        if not test_mode:
            self.t_env += self.t

        cur_returns[0].append(episode_return)
        cur_returns[1].append(smac_reward)
        cur_returns[2].append(task_reward)

        if self.args.test_eval:
            self._log(cur_returns, cur_stats, log_prefix, task_stats)
        if test_mode and (len(self.test_returns[0]) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix, task_stats)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix, task_stats)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix, stats2):
        rewards = ['_','_smac_','_'+self.args.task+'_']
        for idx,return_type in enumerate(returns):
            self.logger.log_stat(prefix + "return" + rewards[idx] + "mean", np.mean(return_type), self.t_env)
            self.logger.log_stat(prefix + "return" + rewards[idx] + "std", np.std(return_type), self.t_env)
            returns[idx].clear()

        for k, v in stats2.items():
            self.logger.log_stat(prefix + k + "_mean", np.mean(v), self.t_env)
            self.logger.log_stat(prefix + k + "_std", np.std(v), self.t_env)
            stats2[k].clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()