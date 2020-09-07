__all__ = ['Monitor', 'get_monitor_files', 'load_results']

import csv
import json
import os
import time
from glob import glob
from typing import Tuple, Dict, Any, List, Optional
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
import pandas
import numpy as np
from stable_baselines3.common import logger

class Monitor(gym.Wrapper):
    """
    A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.
    :param env: (gym.Env) The environment
    :param filename: (Optional[str]) the location to save a log file, can be None for no log
    :param allow_early_resets: (bool) allows the reset of the environment before it is done
    :param reset_keywords: (Tuple[str, ...]) extra keywords for the reset call,
        if extra parameters are needed at reset
    :param info_keywords: (Tuple[str, ...]) extra information to log, from the information return of env.step()
    """

    def __init__(self,
                 env: gym.Env,
                 filename: Optional[str] = None,
                 allow_early_resets: bool = True,
                 reset_keywords: Tuple[str, ...] = (),
                 info_keywords: Tuple[str, ...] = ()):
        super(Monitor, self).__init__(env=env)
        self.t_start = time.time()
        #self.logger_dir = self.logger.get_dir() +'/'+ os.listdir(self.logger.get_dir())[0]
        
            
        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.allow_early_resets = allow_early_resets
        self.rewards = None
        self.needs_reset = True
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        self.current_reset_info = {}  # extra info about the current episode, that was passed in during reset()

    def reset(self, **kwargs) -> np.ndarray:
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True
        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: (np.ndarray) the first observation of the environment
        """
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError("Tried to reset an environment before done. If you want to allow early resets, "
                               "wrap your env with Monitor(env, path, allow_early_resets=True)")
        self.rewards = []
        self.needs_reset = False
        for key in self.reset_keywords:
            value = kwargs.get(key)
            if value is None:
                raise ValueError('Expected you to pass kwarg {} into reset'.format(key))
            self.current_reset_info[key] = value
        return self.env.reset(**kwargs)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return DummyVecEnv.env_method(method_name, *method_args, indices=None, **method_kwargs)



    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        """
        Step the environment with the given action
        :param action: (np.ndarray) the action
        :return: (Tuple[np.ndarray, float, bool, Dict[Any, Any]]) observation, reward, done, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        if done:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
           
            ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_rewards.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if len(self.episode_rewards)<10:
                ep_rew_mean = np.mean(self.episode_rewards)
            else :
                ep_rew_mean = np.mean(self.episode_rewards[-10:])
            
            logger.record("train/average_reward", ep_rew_mean)

            info['episode'] = ep_info
        self.total_steps += 1
        return observation, reward, done, info

    def close(self):
        """
        Closes the environment
        """
        super(Monitor, self).close()

    def get_total_steps(self) -> int:
        """
        Returns the total number of timesteps
        :return: (int)
        """
        return self.total_steps

    def get_episode_rewards(self) -> List[float]:
        """
        Returns the rewards of all the episodes
        :return: ([float])
        """
        return self.episode_rewards

    def get_episode_lengths(self) -> List[int]:
        """
        Returns the number of timesteps of all the episodes
        :return: ([int])
        """
        return self.episode_lengths

    def get_episode_times(self) -> List[float]:
        """
        Returns the runtime in seconds of all the episodes
        :return: ([float])
        """
        return self.episode_times


