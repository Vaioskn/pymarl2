import random
import os
from typing import Tuple, Any, Dict

import numpy as np
import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.wrappers import TimeLimit as GymTimeLimit
from gymnasium.utils.step_api_compatibility import step_api_compatibility

from pymarlzooplus.envs.multiagentenv import MultiAgentEnv


class TimeLimitOvercooked(GymTimeLimit):

    def __init__(self, env, max_episode_steps=None):
        super().__init__(env, max_episode_steps=max_episode_steps)

        assert max_episode_steps is not None, "'max_episode_steps' is None!"
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def timelimit_step(self, action) -> Tuple[Any, float, bool, Dict[str, Any]]:
        assert (self._elapsed_steps is not None), "Cannot call env.step() before calling reset()"

        observation, reward, done, info = step_api_compatibility(self.env.step(action), output_truncation_bool=False)

        self._elapsed_steps += 1
        info["TimeLimit.truncated"] = False  # There is no truncation in Overcooked
        if self._elapsed_steps >= self._max_episode_steps:
            done = True

        return observation, reward, done, info

    def get_elapsed_steps(self):
        return self._elapsed_steps


class ObservationOvercooked(ObservationWrapper):
    """
    Observation wrapper that fixes the order of agents' observations.
    """

    def __init__(self, env):
        super(ObservationOvercooked, self).__init__(env)

        self.observation_space: tuple = env.observation_space.shape
        self.timelimit_env = env
        self.other_agent_idx = None
        self.agent_policy_idx = None

    def observation(self, observation):

        if hasattr(self.timelimit_env, 'get_elapsed_steps'):
            if self.timelimit_env.get_elapsed_steps() == 0:  # Called from reset()
                # Get agents' ids to fix their observations and actions' order
                self.other_agent_idx = observation['other_agent_env_idx']
                self.agent_policy_idx = 1 - self.other_agent_idx
        else:
            raise AttributeError("The 'get_elapsed_steps' method is not implemented in the '_OvercookedWrapper'")

        # Fix the order of observations, 'policy_agent_idx' always corresponds to agent 0
        assert self.agent_policy_idx == 1 - self.other_agent_idx
        assert self.other_agent_idx == observation['other_agent_env_idx']
        observation = (
            observation['both_agent_obs'][self.agent_policy_idx],
            observation['both_agent_obs'][self.other_agent_idx]
        )

        return observation

    def step(self, action) -> Tuple[Any, float, bool, Dict[str, Any]]:
        if hasattr(self.timelimit_env, 'timelimit_step'):
            observation, reward, done, info = self.timelimit_env.timelimit_step(action)
        else:
            raise AttributeError("The 'timelimit_step' method is not implemented in the '_OvercookedWrapper'")

        return self.observation(observation), reward, done, info


OVERCOOKED_KEY_CHOICES = [
    "random3",
    "random0",
    "unident",
    "soup_coordination",
    "small_corridor",
    "simple_tomato",
    "simple_o_t",
    "simple_o",
    "schelling_s",
    "schelling",
    "m_shaped_s",
    "long_cook_time",
    "large_room",
    "forced_coordination_tomato",
    "forced_coordination",
    "cramped_room_tomato",
    "cramped_room_o_3orders",
    "cramped_room",
    "cramped_corridor",
    "counter_circuit_o_1order",
    "counter_circuit",
    "corridor",
    "coordination_ring",
    "centre_objects",
    "centre_pots",
    "asymmetric_advantages",
    "asymmetric_advantages_tomato",
    "bottleneck"
]
OVERCOOKED_REWARD_TYPE_CHOICES = ["shaped", "sparse"]


class _OvercookedWrapper(MultiAgentEnv):

    def __init__(self, key, time_limit=500, seed=1, reward_type="sparse", render=False):

        super().__init__()

        # Check key validity
        assert key in OVERCOOKED_KEY_CHOICES, \
            f"Invalid 'key': {key}! \nChoose one of the following: \n{OVERCOOKED_KEY_CHOICES}"
        # Check time_limit validity
        assert isinstance(time_limit, int), \
            f"Invalid time_limit type: {type(time_limit)}, 'time_limit': {time_limit}, is not 'int'!"
        # Check reward_type validity
        assert reward_type in OVERCOOKED_REWARD_TYPE_CHOICES, \
            f"Invalid 'reward_type': {reward_type}! \nChoose one of the following: \n{OVERCOOKED_REWARD_TYPE_CHOICES}"

        self.key = key
        self._seed = seed  # Just for compatibility since the agents start always from the same position
        self.reward_type = reward_type
        self.render_bool = render

        # Placeholders
        self._obs = None
        self._info = None
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

        # Gymnasium make
        from pymarlzooplus.envs.overcooked_ai.src.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
        from pymarlzooplus.envs.overcooked_ai.src.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
        mdp = OvercookedGridworld.from_layout_name(self.key)
        base_env = OvercookedEnv.from_mdp(mdp, horizon=time_limit)
        self.original_env = gym.make("Overcooked-v0", base_env=base_env, featurize_fn=base_env.featurize_state_mdp)

        # Use the wrappers for handling the time limit and the environment observations properly.
        self.episode_limit = time_limit
        self.n_agents = 2  # Always 2 agents
        self.timelimit_env = TimeLimitOvercooked(self.original_env, max_episode_steps=self.episode_limit)
        self._env = ObservationOvercooked(self.timelimit_env)

        # Define the observation space
        self.observation_space: tuple = self._env.observation_space  # type: ignore[override]

        # Define the action space
        if hasattr(self._env.action_space, 'n'):
            self.action_space = self._env.action_space.n
        else:
            raise AttributeError(f"'n' attribute not found in action space in overcooked environment with key: {key}")

        # Needed for rendering
        import cv2
        self.cv2 = cv2

    def get_print_info(self):
        print_info = self.internal_print_info

        # Clear the internal print info
        self.internal_print_info = None

        return print_info

    def step(self, actions):
        """ Returns reward, terminated, info """

        if self.render_bool is True:
            self.render()

        # Fix the order of actions, 'policy_agent_idx' always corresponds to agent 0
        actions = [int(a) for a in actions]
        if self._env.agent_policy_idx == 1:
            actions = actions[::-1]  # reverse the order

        # Make the environment step
        self._obs, reward, done, self._info = self._env.step(actions)

        if self.reward_type == "shaped":
            assert type(self._info['shaped_r_by_agent']) is list, \
                "'self._info['shaped_r_by_agent']' is not a list! " + \
                f"'self._info['shaped_r_by_agent']': {self._info['shaped_r_by_agent']}"
            reward = sum(self._info['shaped_r_by_agent'])
        # else: the other option is the sum of sparse rewards which is the default 'reward'

        # Keep only 'TimeLimit.truncated' in 'self._info'
        self._info = {"TimeLimit.truncated": self._info["TimeLimit.truncated"]}

        # Handle different cases of 'done'
        if isinstance(done, (list, tuple)):
            done = all(done)
        else:
            assert isinstance(done, bool) and done is True

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

        # Randomness does not affect Overcooked, so we don't pass it to the environment
        if seed is not None:
            self._seed = seed

        self._obs, _ = self._env.reset()

        return self.get_obs(), self.get_state()

    def get_info(self):
        return self._info

    def get_n_agents(self):
        return self.n_agents

    def render(self):
        if self.render_capable is True:
            try:
                image = self._env.render()
                image = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2RGB)
                self.cv2.imshow("Overcooked", image)
                self.cv2.waitKey(1)
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
