import random
from typing import Tuple, Any, Dict, List
import os

import gymnasium as gym
from gymnasium.utils.step_api_compatibility import step_api_compatibility
from gymnasium.wrappers import TimeLimit as GymTimeLimit
from gymnasium import register
import numpy as np

from pymarlzooplus.envs.multiagentenv import MultiAgentEnv


class TimeLimitCT(GymTimeLimit):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env, max_episode_steps=max_episode_steps)

        assert max_episode_steps is not None, "'max_episode_steps' is None!"
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, actions) -> Tuple[Any, List[float], bool, Dict[str, Any]]:
        assert (self._elapsed_steps is not None), "Cannot call env.step() before calling reset()"

        observations, rewards, done, info = step_api_compatibility(self.env.step(actions), output_truncation_bool=False)

        # Fix done
        if isinstance(done, (list, tuple)):
            done = all(done)
        else:
            assert isinstance(done, (bool, int))
            if isinstance(done, int):
                done = bool(done)  # From int to bool

        self._elapsed_steps += 1
        info = {"TimeLimit.truncated": False}

        if self._elapsed_steps >= self._max_episode_steps:
            done = True

        return observations, rewards, done, info


class _CaptureTargetWrapper(MultiAgentEnv):

    def __init__(self, key, seed=1, time_limit=500, render=False, **kwargs):

        super().__init__()

        # Check time_limit validity
        assert isinstance(time_limit, int), \
            f"Invalid time_limit type: {type(time_limit)}, 'time_limit': {time_limit}, is not 'int'!"

        # Fix kwargs, since the environment defines this arg as 'terminate_step'
        kwargs['terminate_step'] = time_limit

        self.render_bool = render
        self.key = key
        self._seed = seed

        # Gymnasium registration
        self.gym_registration()

        # Keep the original environment
        self.original_env = gym.make(f"{key}", **kwargs)

        # Set the seed
        if hasattr(self.original_env.unwrapped, 'seed'):
            self._seed = self.original_env.unwrapped.seed(self._seed)
        else:
            raise AttributeError(f"'seed' attribute not found in the Capture Target environment with key: {key}")

        # Get the timelimit
        if hasattr(self.original_env.unwrapped, 'terminate_step'):
            self.episode_limit = self.original_env.unwrapped.terminate_step
        else:
            raise AttributeError(
                f"'terminate_step' attribute not found in the Capture Target environment with key: {key}")

        # TimeLimit wrapper
        self._env = TimeLimitCT(self.original_env, max_episode_steps=self.episode_limit)

        # Get the number of agents
        if hasattr(self.original_env.unwrapped, 'n_agent'):
            self.n_agents = self.original_env.unwrapped.n_agent
            # Check if there are only 2 agents
            assert self.n_agents == 2
        else:
            raise AttributeError(f"'n_agents' attribute not found in the Capture Target environment with key: {key}")

        # Define the observation space
        if hasattr(self.original_env.unwrapped, 'obs_size'):
            self._obs_size = self.original_env.unwrapped.obs_size[0]
            # Check if obs_size is the same for every agent
            assert all([self._obs_size == obs_size for obs_size in self.original_env.unwrapped.obs_size])
        else:
            raise AttributeError(
                "'obs_size' attribute not found in 'original_env.unwrapped' "
                f"of the Capture Target environment with key: {key}"
            )

        # Define the action space
        if hasattr(self.original_env.unwrapped, 'n_action'):
            self.action_space = self.original_env.unwrapped.n_action[0]
            # Check if n_action is the same for every agent
            assert all([self.action_space == n_action for n_action in self.original_env.unwrapped.n_action])
        else:
            raise AttributeError(
                "'n_action' attribute not found in 'original_env.unwrapped' "
                f"of the Capture Target environment with key: {key}"
            )

        # Placeholders
        self._obs = None
        self._info = {"TimeLimit.truncated": False}
        self.internal_print_info = None

        # Check the consistency between the 'render_bool' and the display capabilities of the machine
        self.render_capable = True
        if self.render_bool is True and 'DISPLAY' not in os.environ:
            self.render_bool = False
            self.internal_print_info = (
                "\n\n###########################################################"
                "\nThe 'render' is set to 'False' due to the lack of display capabilities!"
                "\n###########################################################\n"
            )
            self.render_capable = False

    @staticmethod
    def gym_registration():
        register(
            id="CaptureTarget-6x6-1t-2a-v0",
            entry_point="pymarlzooplus.envs.capture_target.src.capture_target_ai_py.environment:CaptureTarget",
            kwargs={
                "n_target": 1,
                "n_agent": 2,
                "grid_dim": [6, 6]
            },
        )

    def get_print_info(self):
        print_info = self.internal_print_info

        # Clear the internal print info
        self.internal_print_info = None

        return print_info

    def step(self, actions):
        """ Returns reward, terminated, info """

        if self.render_bool is True:
            self.render()

        # From torch.tensor to int
        actions = [int(a) for a in actions]

        # Make the environment step
        self._obs, rewards, done, info = self._env.step(actions)
        # From numpy to tuple
        self._obs = tuple(self._obs)

        # Add all rewards together. 'rewards' is a list
        rewards = sum(rewards)

        return float(rewards), done, {}

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self._obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self._obs_size

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """ Returns the flatted shape of the state"""
        return self.n_agents * self._obs_size

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return self.action_space * [1]  # 1 indicates availability of actions

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.action_space

    def sample_actions(self):
        return random.choices(range(0, self.get_total_actions()), k=self.n_agents)

    def reset(self, seed=None):
        """ Returns initial observations and states"""

        # Control seed
        if seed is not None:
            if hasattr(self.original_env.unwrapped, 'seed'):
                self._seed = self.original_env.unwrapped.seed(seed)
            else:
                raise AttributeError(f"'seed' attribute not found in the Capture Target environment with key: {self.key}")

        self._obs, _ = self._env.reset()
        # From numpy to tuple
        self._obs = tuple(self._obs)

        return self.get_obs(), self.get_state()

    def get_info(self):
        return self._info

    def get_n_agents(self):
        return self.n_agents

    def render(self):
        if self.render_capable is True:
            try:
                self._env.render()
            except (Exception, SystemExit) as e:
                self.internal_print_info = (
                    "\n\n###########################################################"
                    f"\nError during rendering: \n\n{e}"
                    f"\n\nRendering will be disabled to continue the training."
                    "\n###########################################################\n"
                )
                self.render_capable = False

    def close(self):
        self._env.close()

    def seed(self):
        if hasattr(self.original_env.unwrapped, 'seed'):
            return self.original_env.unwrapped.seed()
        else:
            raise AttributeError(f"'seed' attribute not found in the Capture Target environment with key: {self.key}")

    def save_replay(self):
        pass

    @staticmethod
    def get_stats():
        return {}
