from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th
import sys

# Get the multitask functions
sys.path.insert(0,'/home/aroman/TFM/Código/Proyecto_TFM/Proyecto_TFM/src/Multi_task')

from rewards_multitask import reward_task_objetive, reward_task_survive, reward_task_kill, reward_scalarization, check_ally_death

# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        env_args = [self.args.env_args.copy() for _ in range(self.batch_size)]
        for i in range(self.batch_size):
            env_args[i]["seed"] += i

        self.ps = [Process(target=env_worker, args=(worker_conn, CloudpickleWrapper(partial(env_fn, **env_arg)),args))
                            for env_arg, worker_conn in zip(env_args, self.worker_conns)]

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0

        self.t_env = 0

        self.train_returns = [[] for i in range(3)]
        self.test_returns = [[] for i in range(3)]
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": [],
        }
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        self.reset()

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        smac_returns = [0 for _ in range(self.batch_size)]
        task_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        dist = [0 for _ in range(self.batch_size)]
        task_reach = [0 for _ in range(self.batch_size)]
        death = [0 for _ in range(self.batch_size)]
        number_kill = [0 for _ in range(self.batch_size)]
        survive = [0 for _ in range(self.batch_size)]
        number_death = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        while True:

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1)
            }
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated: # We produced actions for this env
                    if not terminated[idx]: # Only send the actions to the env if it hasn't terminated
                        if self.args.task == 'objetive':
                            parent_conn.send(("step", (cpu_actions[action_idx],task_returns[idx],dist[idx],self.t)))
                        elif self.args.task == 'survive':
                            parent_conn.send(("step", (cpu_actions[action_idx],number_death[idx])))
                        else:
                            parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1 # actions is not a list over every env

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            post_transition_data = {
                "reward": [],
                "terminated": []
                }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": []
            }

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()

                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))
                    if self.args.task == "objetive":
                        task_returns[idx] += data["task_reward"]
                        smac_returns[idx] += data["smac_reward"]
                        episode_returns[idx] += data["reward"]
                        dist[idx] = data["dist"]
                        task_reach[idx] = data["task_reach"]
                    elif self.args.task == "kill": 
                        task_returns[idx] += data["task_reward"]
                        smac_returns[idx] += data["smac_reward"]
                        episode_returns[idx] += data["reward"]
                        number_kill[idx] = data["number_kill"]
                        death[idx] = data["death"]
                    elif self.args.task == "survive":
                        number_death[idx] = data["number_death"]
                        episode_returns[idx] += data["reward"]

                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats",None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos
        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        # # Add task and extrinsic rewards to metrics
        # if self.args.task == 'objetive':
        #     cur_stats['dist'] = dist
        #     cur_stats['objetive_reach'] = task_reach
        # elif self.args.task == 'kill':
        #     cur_stats['number_kill'] = number_kill
        #     cur_stats['kill'] = death
        # elif self.args.task == 'survive':
        #     cur_stats['survive'] = survive
        #     cur_stats['number_death'] = number_death

        cur_returns[0].extend(episode_returns)
        cur_returns[1].extend(smac_returns)
        cur_returns[2].extend(task_returns)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        rewards = ['_','_smac_','_'+self.args.task+'_']
        for idx,return_type in enumerate(returns):
            self.logger.log_stat(prefix + "return" + rewards[idx] + "mean", np.mean(return_type), self.t_env)
            self.logger.log_stat(prefix + "return" + rewards[idx] + "std", np.std(return_type), self.t_env)
            returns[idx].clear()
            
        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()


def env_worker(remote, env_fn, args):
    # Make environment
    env = env_fn.x()
    obj = create_obj(env,args)
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            # obj = create_obj(env,args)
            if obj.args.task == 'objetive':
                actions, task_reward, dist_prev, time = data
                # Take a step in the environment
                reward, terminated, env_info = env.step(actions)
                # Check if the multi-objetive option is set and compute 
                # the aditional reward and
                # the total reward scalarization
                reward_objetive,terminated_task,dist = reward_task_objetive(obj,task_reward,terminated,dist_prev,time)
                if terminated_task:
                    task_reach = 1
                else:
                    task_reach = 0
                terminated = terminated_task
                rewards = reward_scalarization(obj,reward,reward_objetive)
                reward = rewards[0]
                smac_reward = rewards[1]
                task_reward = rewards[2]
                # Return the observations, avail_actions and state to make the next action
                state = env.get_state()
                avail_actions = env.get_avail_actions()
                obs = env.get_obs()
                remote.send({
                    # Data for the next timestep needed to pick an action
                    "state": state,
                    "avail_actions": avail_actions,
                    "obs": obs,
                    # Rest of the data for the current timestep
                    "reward": reward,
                    "terminated": terminated,
                    "info": env_info,
                    # Multitasking info
                    "smac_reward": smac_reward,
                    "task_reward": task_reward,
                    "task_reach": task_reach,
                    "dist": dist
                })
            elif obj.args.task == 'kill':
                actions = data
                # Take a step in the environment
                reward, terminated, env_info = env.step(actions)
                # Check if the multi-objetive option is set and compute 
                # the aditional reward and
                # the total reward scalarization
                reward_kill, death, number_kill = reward_task_kill(obj)
                rewards = reward_scalarization(obj,reward,reward_kill)
                reward = rewards[0]
                smac_reward = rewards[1]
                task_reward = rewards[2]    
                # Return the observations, avail_actions and state to make the next action
                state = env.get_state()
                avail_actions = env.get_avail_actions()
                obs = env.get_obs()
                remote.send({
                    # Data for the next timestep needed to pick an action
                    "state": state,
                    "avail_actions": avail_actions,
                    "obs": obs,
                    # Rest of the data for the current timestep
                    "reward": reward,
                    "terminated": terminated,
                    "info": env_info,
                    # Multitasking info
                    "smac_reward": smac_reward,
                    "task_reward": reward_kill,
                    "death": death,
                    "number_kill": number_kill
                })
            elif obj.args.task == "survive" and not terminated:
                actions, number_death = data
                # Take a step in the environment
                reward, terminated, env_info = env.step(actions)
                # Check if the multi-objetive option is set and compute 
                # the aditional reward and
                # the total reward scalarization
                # if not terminated:
                number_death = check_ally_death(obj,number_death)
                state = env.get_state()
                avail_actions = env.get_avail_actions()
                obs = env.get_obs()
                remote.send({
                    # Data for the next timestep needed to pick an action
                    "state": state,
                    "avail_actions": avail_actions,
                    "obs": obs,
                    # Rest of the data for the current timestep
                    "reward": reward,
                    "terminated": terminated,
                    "info": env_info,
                    # Multitasking info
                    "number_death": number_death
                })
                # elif terminated:
                #     reward_survive, survive = reward_task_survive(obj,number_death)
                #     rewards = reward_scalarization(obj,reward,reward_survive)
                #     reward = rewards[0]
                #     smac_reward = rewards[1]
                #     task_reward = rewards[2]
                #     remote.send({
                #         # Multitasking info
                #         "reward": reward,
                #         "smac_reward": smac_reward,
                #         "task_reward": task_reward,
                #         "survive": survive
                #     })
            else:
                state = env.get_state()
                avail_actions = env.get_avail_actions()
                obs = env.get_obs()
                remote.send({
                    # Data for the next timestep needed to pick an action
                    "state": state,
                    "avail_actions": avail_actions,
                    "obs": obs,
                    # Rest of the data for the current timestep
                    "reward": reward,
                    "terminated": terminated,
                    "info": env_info
                })
        elif cmd == "reset":
            env.reset()
            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs()
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class create_obj():
    def __init__(self,x,y):
        self.env = x
        self.args = y