import random
import os
from typing import Tuple, Any, Dict, List

import numpy as np
import gymnasium as gym
from gymnasium.utils.step_api_compatibility import step_api_compatibility
from gymnasium.wrappers import TimeLimit as GymTimeLimit

from pymarlzooplus.envs.multiagentenv import MultiAgentEnv


class TimeLimitPressurePlate(GymTimeLimit):
    """Wraps the original environment and adds the extra var "elapsed_time" to keep track of when an episode starts"""

    def __init__(self, env, max_episode_steps=None):
        super().__init__(env, max_episode_steps=max_episode_steps)

        assert max_episode_steps is not None, "'max_episode_steps' is None!"
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action) -> Tuple[Any, List[float], List[bool], Dict[str, Any]]:
        assert (self._elapsed_steps is not None), "Cannot call env.step() before calling reset()"
        observations, rewards, terminations, infos = step_api_compatibility(
            self.env.step(action),
            output_truncation_bool=False
        )

        self._elapsed_steps += 1
        infos["TimeLimit.truncated"] = False  # fake var, there is no truncation in PressurePlate
        if self._elapsed_steps >= self._max_episode_steps:
            terminations = [True] * len(terminations)

        return observations, rewards, terminations, infos


PRESSUREPLATE_KEY_CHOICES = [
    "pressureplate-linear-4p-v0",
    "pressureplate-linear-5p-v0",
    "pressureplate-linear-6p-v0"
]

PRESSUREPLATE_N_AGENTS_CHOICES = [4, 5, 6]


class _PressurePlateWrapper(MultiAgentEnv):

    def __init__(self, key, seed=1, time_limit=500, render=False):

        super().__init__()

        # Check key validity
        assert key in PRESSUREPLATE_KEY_CHOICES, \
            f"Invalid 'key': {key}! \nChoose one of the following: \n{PRESSUREPLATE_KEY_CHOICES}"
        # Check time_limit validity
        assert isinstance(time_limit, int), \
            f"Invalid time_limit type: {type(time_limit)}, 'time_limit': {time_limit}, is not 'int'!"

        self.key = key
        self._seed = seed
        self.render_bool = render

        # Placeholders
        self.original_env = None
        self.episode_limit = None
        self._env = None
        self._obs = None
        self._info = None
        self.observation_space = None
        self.action_space = None
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

        ## We import the environment to do the "gymnasium make" base env sourced by gym.make with all its args
        from pymarlzooplus.envs.pressureplate_ai.pressureplate.environment import PressurePlate
        self.original_env = gym.make(f"{key}")
        if hasattr(self.original_env.unwrapped, 'seed'):
            self._seed = self.original_env.unwrapped.seed(self._seed)
        else:
            raise AttributeError(f"'seed' attribute not found in the Pressure Plate environment with key: {key}")

        # Get the number of agents
        if hasattr(self.original_env.unwrapped, 'n_agents'):
            self.n_agents = self.original_env.unwrapped.n_agents
        else:
            raise AttributeError(f"'n_agents' attribute not found in the Pressure Plate environment with key: {key}")

        # Use the TimiLimit wrapper for handling the time limit properly.
        self.episode_limit = time_limit
        self._env = TimeLimitPressurePlate(self.original_env, max_episode_steps=self.episode_limit)

        # Define the observation space
        if hasattr(self._env.observation_space, 'spaces'):
            self.observation_space = self._env.observation_space.spaces[0].shape
        else:
            raise AttributeError(
                f"'spaces' attribute not found in the observation space of the Pressure Plate environment with key: {key}"
            )
        # Define the action space
        if hasattr(self._env.action_space, 'spaces'):
            self.action_space = self._env.action_space.spaces[0].n
        else:
            raise AttributeError(
                f"'spaces' attribute not found in the action space of the Pressure Plate environment with key: {key}"
            )
        # Placeholders
        self._obs = None
        self._info = None

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
        self._obs, rewards, terminations, self._info = self._env.step(actions)

        # Add all rewards together. 'rewards' is a list
        reward = sum(rewards)

        # Keep only 'TimeLimit.truncated' in 'self._info'
        self._info = {"TimeLimit.truncated": self._info["TimeLimit.truncated"]}

        # The episode ends when all agents have reached their positions ("terminations" are all True) or
        # "self._elapsed_steps >= self._max_episode_steps" is True
        done = all(terminations)

        return float(reward), done, {}

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self._obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.observation_space[0]

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """ Returns the shape of the state """

        assert len(self.observation_space) == 1, \
            f"'self.observation_space' has not only one dimension! \n'self.observation_space': {self.observation_space}"

        return self.n_agents * self.observation_space[0]

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)

        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id (both agents have the same action space) """
        return self.action_space * [1]  # 1 indicates availability of action

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return int(self.action_space)

    def sample_actions(self):
        return random.choices(range(0, self.get_total_actions()), k=self.n_agents)

    def reset(self, seed=None):
        """ Returns initial observations and states """

        # Control seed
        # NOTE: Pressure Plate is not affected by the randomness of the seed
        if seed is not None:
            if hasattr(self.original_env.unwrapped, 'seed'):
                self._seed = self.original_env.unwrapped.seed(seed)
            else:
                raise AttributeError(f"'seed' attribute not found in the Pressure Plate environment with key: {self.key}")

        self._obs, _ = self._env.reset()
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
        return self._seed

    def save_replay(self):
        pass

    @staticmethod
    def get_stats():
        return {}
