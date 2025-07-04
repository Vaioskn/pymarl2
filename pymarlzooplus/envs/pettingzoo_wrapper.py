import os

import numpy as np
import torch
from gymnasium.utils import seeding
import cv2

from pymarlzooplus.envs.multiagentenv import MultiAgentEnv
from pymarlzooplus.utils.image_encoder import ImageEncoder
from pymarlzooplus.utils.env_utils import pettingzoo_make


class TimeLimitPZ(object):
    """Custom TimeLimit wrapper for PettingZoo environments."""
    def __init__(self, env, key, max_episode_steps):

        assert (
                isinstance(max_episode_steps, int) and max_episode_steps > 0
        ), f"Expect the `max_episode_steps` to be positive, actually: {max_episode_steps}"

        self.env = env
        self._key = key
        self._max_episode_steps = max_episode_steps

        # Placeholders
        self._elapsed_steps = None
        self._obs_wrapper = None

    def timelimit_step(self, actions):
        assert (self._elapsed_steps is not None), "Cannot call env.step() before calling reset()"

        infos = {}

        if self._key == "entombed_cooperative_v3":

            # In this step, agent 1 is moving
            previous_observations_, previous_rewards_, previous_terminations_, previous_truncations_, _ = \
                self.env.step(actions)
            previous_terminations__all = any([termination for termination in previous_terminations_.values()])
            previous_truncations__all = any([truncation for truncation in previous_truncations_.values()])

            if previous_terminations__all is False and previous_truncations__all is False:

                # In this step, agent 2 is moving
                observations_, rewards_, terminations_, truncations_, _ = self.env.step(actions)
                terminations__all = any([termination for termination in terminations_.values()])
                truncations__all = any([truncation for truncation in truncations_.values()])

                if terminations__all is False and truncations__all is False:

                    ## Perform no action 2 times in order to sync obs and actions
                    no_actions = {'first_0': 0, 'second_0': 0}

                    # First no action
                    previous_observations, previous_rewards, previous_terminations, previous_truncations, _ = \
                        self.env.step(no_actions)
                    previous_obs = list(previous_observations.values())[0]
                    previous_terminations_all = any([termination for termination in previous_terminations.values()])
                    previous_truncations_all = any([truncation for truncation in previous_truncations.values()])

                    if previous_terminations_all is False and previous_truncations_all is False:

                        # Second no action
                        observations, rewards, terminations, truncations, infos = self.env.step(no_actions)
                        current_obs = list(observations.values())[1]

                        # Get the combined obs
                        observations = \
                            self._obs_wrapper.entombed_cooperative_v3_get_combined_images(previous_obs, current_obs)

                        rewards = {
                            reward_key: sum([reward, reward_, previous_reward, previous_reward_])
                            for (reward_key, reward, reward_, previous_reward, previous_reward_) in
                            zip(rewards.keys(), rewards.values(), rewards_.values(), previous_rewards.values(), previous_rewards_.values())
                        }

                        self._elapsed_steps += 4
                    else:  # Third step termination case
                        # In this case, we don't really care about observations
                        # since it's the last step (due to termination)
                        observations = previous_observations
                        rewards = {
                            reward_key: sum([reward_, previous_reward, previous_reward_])
                            for (reward_key, reward_, previous_reward, previous_reward_) in
                            zip(rewards_.keys(), rewards_.values(), previous_rewards.values(), previous_rewards_.values())
                        }
                        terminations = previous_terminations
                        truncations = previous_truncations
                        self._elapsed_steps += 3
                else:  # Second step termination case
                    # In this case, we don't really care about observations
                    # since it's the last step (due to termination)
                    observations = observations_
                    rewards = {
                        reward_key: sum([reward_, previous_reward_]) for (reward_key, reward_, previous_reward_) in
                        zip(rewards_.keys(), rewards_.values(), previous_rewards_.values())
                    }
                    terminations = terminations_
                    truncations = truncations_
                    self._elapsed_steps += 2
            else:  # First step termination case
                # In this case, we don't really care about observations
                # since it's the last step (due to termination)
                observations = previous_observations_
                rewards = previous_rewards_
                terminations = previous_terminations_
                truncations = previous_truncations_
                self._elapsed_steps += 1

        elif self._key == "space_invaders_v2":
            # After extensive investigation, we found out that the "move" actions should be applied at the first step
            # and the "fire" actions at the second step, in order to apply the actions consistently.
            move_actions = {
                'first_0': actions['first_0'] if actions['first_0'] != 1 else 0,
                'second_0': actions['second_0'] if actions['second_0'] != 1 else 0
            }
            fire_actions = {
                'first_0': actions['first_0'] if actions['first_0'] == 1 else 0,
                'second_0': actions['second_0'] if actions['second_0'] == 1 else 0
            }
            # Perform the decided actions in order to get the state which is not full due to flickering
            previous_observations, previous_rewards, previous_terminations, previous_truncations, _ = \
                self.env.step(move_actions)
            previous_obs = list(previous_observations.values())[0]
            previous_terminations_all = any([termination for termination in previous_terminations.values()])
            previous_truncations_all = any([truncation for truncation in previous_truncations.values()])

            if previous_terminations_all is False and previous_truncations_all is False:
                # Perform no action and get the next state which is not full due to flickering
                observations, rewards, terminations, truncations, infos = self.env.step(fire_actions)
                current_obs = list(observations.values())[1]

                # Combine the two states to get the full state after applying the actions
                observations = self._obs_wrapper.space_invaders_v2_get_combined_images(previous_obs, current_obs)

                rewards = {
                    reward_key: sum([reward, previous_reward]) for (reward_key, reward, previous_reward) in
                    zip(rewards.keys(), rewards.values(), previous_rewards.values())
                }

                self._elapsed_steps += 2

            else:  # First step termination case
                # In this case, we don't really care about observations
                # since it's the last step (due to termination)
                observations = previous_observations
                rewards = previous_rewards
                terminations = previous_terminations
                truncations = previous_truncations
                self._elapsed_steps += 1

        else:
            observations, rewards, terminations, truncations, infos = self.env.step(actions)
            self._elapsed_steps += 1

        infos["TimeLimit.truncated"] = any([truncation for truncation in truncations.values()])

        if self._elapsed_steps >= self._max_episode_steps:
            terminations = {key: True for key in terminations.keys()}

        return observations, rewards, terminations, truncations, infos

    def reset(self, seed=None, options=None):
        self._elapsed_steps = 0
        return self.env.reset(seed=seed, options=options)

    def close(self):
        return self.env.close()

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space


class ObservationPZ(object):
    """
    Custom Observation wrapper for converting images to vectors (using a pretrained image encoder) or
    for preparing images to be fed to a CNN.
    """

    def __init__(
            self,
            env,
            partial_observation,
            trainable_cnn,
            image_encoder,
            image_encoder_batch_size,
            image_encoder_use_cuda,
            centralized_image_encoding,
            given_observation_space
    ):

        # Keep the parent environment
        self.env = env

        # Keep the original PettingZoo environment
        self.original_env = env.env

        # Keep the timelimit environment
        self.timelimit_env = env

        # Initialize 'ImageEncoder' and get useful attributes
        self.image_encoder = ImageEncoder(
            "env",
            centralized_image_encoding,
            trainable_cnn,
            image_encoder,
            image_encoder_batch_size,
            image_encoder_use_cuda
        )
        self.print_info = self.image_encoder.print_info
        self.observation_space = self.image_encoder.observation_space

        self.partial_observation = partial_observation
        self.original_observation_space = self.original_env.observation_space(self.original_env.possible_agents[0])
        self.original_observation_space_shape = self.original_observation_space.shape
        self._is_image = len(self.original_observation_space_shape) == 3 and self.original_observation_space_shape[2] == 3
        assert self._is_image, f"Only images are supported, shape: {self.original_observation_space_shape}"

        self.given_observation_space = given_observation_space
        if given_observation_space is not None:
            self.observation_space = given_observation_space
        assert not (given_observation_space is None and centralized_image_encoding is True)
        assert not (given_observation_space is not None and centralized_image_encoding is False)

    def step(self, actions):
        if hasattr(self.timelimit_env, 'timelimit_step'):
            observations, rewards, terminations, truncations, infos = self.timelimit_env.timelimit_step(actions)
            return self.observation(observations), rewards, terminations, truncations, infos
        raise AttributeError("The 'timelimit_step' method is not implemented in the 'ObservationPZ'")

    def observation(self, observations):
        return self.image_encoder.observation(observations)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info

    @staticmethod
    def replace_color(image_, target_color, replacement_color):
        # Find all pixels matching the target color
        matches = np.all(image_ == target_color, axis=-1)
        # Replace these pixels with the replacement color
        image_[matches] = replacement_color

        return image_

    def space_invaders_v2_get_combined_images(self, image_a, image_b, sensitivity=0):
        # We should remove the red ship, and the two agents from image A in order to get
        # just their final position from image B, otherwise artifacts will be created due
        # to the minor movements of these objects.
        red_ship_rgb_values = [181, 83, 40]
        agent_1_rgb_values = [50, 132, 50]
        agent_2_rgb_values = [162, 134, 56]
        image_a = self.replace_color(image_a, red_ship_rgb_values, [0, 0, 0])
        image_a = self.replace_color(image_a, agent_1_rgb_values, [0, 0, 0])
        image_a = self.replace_color(image_a, agent_2_rgb_values, [0, 0, 0])
        # Calculate the absolute difference between images
        diff = cv2.absdiff(image_a, image_b)
        # Convert the difference to grayscale in order to handle a single threshold for all channels
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        # Mask for common areas: where the difference is less than or equal to sensitivity
        common_mask = np.where(diff_gray <= sensitivity, 255, 0).astype(np.uint8)
        # Mask for differences: where the difference is greater than sensitivity
        difference_mask = np.where(diff_gray > sensitivity, 255, 0).astype(np.uint8)
        # Create a 3-channel mask for common and difference areas
        common_mask_3channel = cv2.cvtColor(common_mask, cv2.COLOR_GRAY2RGB)
        difference_mask_3channel = cv2.cvtColor(difference_mask, cv2.COLOR_GRAY2RGB)
        # Extract common areas using common mask
        common_areas = cv2.bitwise_and(image_a, common_mask_3channel)
        # Extract differences from both images
        differences_from_a = cv2.bitwise_and(image_a, difference_mask_3channel)
        differences_from_b = cv2.bitwise_and(image_b, difference_mask_3channel)
        # Combine common areas with differences from both images
        combined_image = cv2.add(common_areas, differences_from_a)
        combined_image = cv2.add(combined_image, differences_from_b)
        # Create partial obs of agent 1 by removing agent 2 from the combined image
        agent_1_obs_ = self.replace_color(combined_image.copy(), agent_2_rgb_values, [0, 0, 0])
        # Create partial obe of agent 2 by removing agent 1 from the combined image
        agent_2_obs_ = self.replace_color(combined_image.copy(), agent_1_rgb_values, [0, 0, 0])

        if self.partial_observation is True:
            obs = {'first_0': agent_1_obs_, 'second_0': agent_2_obs_}
        else:
            obs = {'first_0': combined_image.copy(), 'second_0': combined_image.copy()}

        return obs

    def entombed_cooperative_v3_get_combined_images(self, image_a, image_b):
        # Define the RGB values of agent 1
        agent_1_rgb_values = np.array([232, 232, 74])
        # Define the RGB values of agent 2
        agent_2_rgb_values = np.array([197, 124, 238])
        # Find where image A has the specific RGB values of agent 1
        mask = np.all(image_a == agent_1_rgb_values, axis=-1)
        # Find where image B has the specific RGB value of agent 1
        mask_ = np.all(image_b == agent_1_rgb_values, axis=-1)
        # Find which is the image that illustrates agent 1
        if mask_.sum() > mask.sum():
            image_b = image_a
            mask = mask_
        # Replace the corresponding values in image B where the mask is True (that is where agent 1 is located)
        combined_image = image_b.copy()
        combined_image[mask] = agent_1_rgb_values
        # Create partial obs of agent 1 by removing agent 2 from the combined image
        black_rgb_values = [0, 0, 0]
        agent_1_obs_ = self.replace_color(combined_image.copy(), agent_1_rgb_values, black_rgb_values)
        # Create partial obe of agent 2 by removing agent 1 from the combined image
        agent_2_obs_ = self.replace_color(combined_image.copy(), agent_2_rgb_values, black_rgb_values)

        if self.partial_observation is True:
            obs = {'first_0': agent_1_obs_, 'second_0': agent_2_obs_}
        else:
            obs = {'first_0': combined_image.copy(), 'second_0': combined_image.copy()}

        return obs

    def close(self):
        return self.env.close()

    @property
    def action_space(self):
        return self.env.action_space


class _PettingZooWrapper(MultiAgentEnv):
    def __init__(
            self,
            key,
            time_limit=None,
            seed=1,
            render_mode="rgb_array",
            partial_observation=False,
            trainable_cnn=False,
            image_encoder="ResNet18",
            image_encoder_batch_size=1,
            image_encoder_use_cuda=False,
            centralized_image_encoding=False,
            kwargs="",
            given_observation_space=None
    ):

        assert (partial_observation is False) or (key in ["entombed_cooperative_v3", "space_invaders_v2"]), \
            (
                "'partial_observation' should be False when the selected game is other than "
                "'entombed_cooperative_v3' or 'space_invaders_v2'!"
            )

        super().__init__()

        self.key = key
        self.max_cycles = time_limit
        self._seed = seed
        self.render_mode = render_mode
        self.partial_observation = partial_observation
        self.trainable_cnn = trainable_cnn
        self._image_encoder = image_encoder
        self.image_encoder_batch_size = image_encoder_batch_size
        self.image_encoder_use_cuda = image_encoder_use_cuda
        self.centralized_image_encoding = centralized_image_encoding
        self._kwargs = kwargs
        self.given_observation_space = given_observation_space

        # Placeholders
        self.kwargs = None
        self.original_env = None
        self._env = None  # Observation wrapper
        self.__env = None  # TimeLimit wrapper
        self._obs = None
        self._info = None
        self.observation_space = None
        self._common_observation_space = None
        self.original_observation_space = None
        self._is_image = None
        self.action_space = None
        self._agent_prefix = None
        self.sum_rewards = None
        self.np_random = None
        self.step_function = None
        self.internal_print_info = None

        # Check the consistency between the 'render_mode' and the display capabilities of the machine
        self.render_capable = True
        if self.render_mode == "human" and 'DISPLAY' not in os.environ:
            self.render_mode = "rgb_array"
            self.internal_print_info = (
                "\n\n###########################################################"
                "\nThe 'render_mode' is set to 'rgb_array' due to the lack of display capabilities!"
                "\n###########################################################\n"
            )
            self.render_capable = False

        # Define the keys for fully cooperative and classic tasks
        self.fully_cooperative_task_keys = [
            "pistonball_v6",
            "cooperative_pong_v5",
            "entombed_cooperative_v3",
            "space_invaders_v2"
        ]
        self.classic_task_keys = [
            "chess_v6",
            "connect_four_v3",
            "gin_rummy_v4",
            "go_v5",
            "hanabi_v5",
            "leduc_holdem_v4",
            "rps_v2",
            "texas_holdem_no_limit_v6",
            "texas_holdem_v4",
            "tictactoe_v3"
        ]

        # Environment
        self.set_environment(
            self.key,
            self.max_cycles,
            self.render_mode,
            self.partial_observation,
            self.trainable_cnn,
            self._image_encoder,
            self.image_encoder_batch_size,
            self.image_encoder_use_cuda,
            self.centralized_image_encoding,
            self._kwargs,
            self.given_observation_space
        )

    def set_environment(
            self,
            key,
            max_cycles,
            render_mode,
            partial_observation,
            trainable_cnn,
            image_encoder,
            image_encoder_batch_size,
            image_encoder_use_cuda,
            centralized_image_encoding,
            kwargs,
            given_observation_space
    ):

        # Convert list of kwargs to dictionary
        self.kwargs = kwargs
        self.get_kwargs(max_cycles, render_mode)

        # Assign value to 'self.sum_rewards' based on the env key
        self.sum_rewards = True
        if key not in self.fully_cooperative_task_keys:
            # Only these environments refer to full cooperation, thus we can sum the rewards.
            self.sum_rewards = False

        # Define the environment
        self.original_env = pettingzoo_make(key, self.kwargs)

        # In the case of classic environments, return the original PettingZoo environment
        if self.key in self.classic_task_keys:
            return

        # Define episode horizon
        if hasattr(self.original_env.unwrapped, 'max_cycles'):
            self.episode_limit = self.original_env.unwrapped.max_cycles
        else:
            assert hasattr(self.original_env.unwrapped.env, 'max_cycles')
            self.episode_limit = self.original_env.unwrapped.env.max_cycles
            assert self.episode_limit > 1, "self.episode_limit should be > 1: {self.episode_limit}"

        # Define the number of agents
        self.n_agents = self.original_env.max_num_agents
        assert (self.n_agents == 2) or (key not in ["entombed_cooperative_v3", "space_invaders_v2"]), \
            (
                "Only 2 agents have been considered for 'entombed_cooperative_v3' and 'space_invaders_v2'! "
                "'n_agents': {}".format(self.n_agents)
            )

        # Define TimeLimit wrapper
        self.__env = TimeLimitPZ(
            self.original_env,
            key,
            max_episode_steps=self.episode_limit
        )
        # In the case of "entombed_cooperative_v3" and "space_invaders_v2" games,
        # fix the "self.episode_limit" after passing it to the "TimeLimitPZ"
        if key == "entombed_cooperative_v3":
            # At each timestep, we apply 4 pettingzoo timesteps,
            # in order to synchronize actions and obs
            assert self.episode_limit % 4 == 0, (
                "When 'entombed_cooperative_v3' is used the specified 'episode_limit' should be divisible by 4: {}"
                .format(self.episode_limit)
            )
            self.episode_limit = int(self.episode_limit / 4)
        elif key == "space_invaders_v2":
            # At each timestep, we apply 2 pettingzoo timesteps,
            # in order to synchronize actions and obs
            assert self.episode_limit % 2 == 0, (
                "When 'space_invaders_v2' is used the specified 'episode_limit' should be divisible by 2: {}"
                .format(self.episode_limit)
            )
            self.episode_limit = int(self.episode_limit / 2)

        # Define Observation wrapper
        self.original_observation_space = self.__env.observation_space(self.original_env.possible_agents[0])
        self._common_observation_space = all(
            [
                self.original_observation_space == self.__env.observation_space(
                    self.original_env.possible_agents[agent_id]
                )
                for agent_id in range(self.n_agents)
            ]
        )
        self._is_image = len(self.original_observation_space.shape) == 3 and self.original_observation_space.shape[2] == 3
        if self._is_image is True:
            self._env = ObservationPZ(
                self.__env,
                partial_observation,
                trainable_cnn,
                image_encoder,
                image_encoder_batch_size,
                image_encoder_use_cuda,
                centralized_image_encoding,
                given_observation_space
            )
            self.__env._obs_wrapper = self._env
            # Define the observation space
            self.observation_space = self._env.observation_space
            # Define the step function
            self.step_function = self._env.step
        else:
            # Define the observation space
            if self._common_observation_space is False:
                self.original_observation_space = self.original_env.observation_spaces  # dictionary
            self._env = self.__env  # Just for consistency
            self.observation_space = self.original_observation_space
            # Define the step function
            self.step_function = self._env.timelimit_step

        self._obs = None
        self._info = None

        ## Define the action space
        tmp_action_spaces_list = [
            self.original_env.action_space(possible_agent) for possible_agent in self.original_env.possible_agents
        ]
        tmp_action_space = tmp_action_spaces_list[0]
        if key in self.fully_cooperative_task_keys:
            # Check that all agents have the same action space only for fully cooperation tasks
            assert not isinstance(tmp_action_space, tuple), f"tmp_action_space: {tmp_action_space}"
            for agent_idx, tmp_action_space_ in enumerate(tmp_action_spaces_list[1:], start=1):
                assert tmp_action_space_ == tmp_action_space, (
                    "Difference in action spaces found between agents:\n"
                    f"agent=0 - action_space={tmp_action_space}, agent={agent_idx} - action_space={tmp_action_space_}"
                )
            self.action_space = tmp_action_space.n.item()
        else:
            self.action_space = tmp_action_space
        self._agent_prefix = [agent_prefix for agent_prefix in self.original_env.possible_agents]

        # Create the seed object to control the randomness of the environment reset()
        self.np_random, self._seed = seeding.np_random(self._seed)

    def get_kwargs(self, max_cycles, render_mode):
        if (
                isinstance(self.kwargs, (list, tuple)) or
                (isinstance(self.kwargs, str) and len(self.kwargs) > 0 and isinstance(eval(self.kwargs), tuple))
        ):
            if not isinstance(self.kwargs, (list, tuple)) and isinstance(eval(self.kwargs), (tuple, list)):
                self.kwargs = eval(self.kwargs)
            if isinstance(self.kwargs[0], (list, tuple)):
                # Convert the list to dict
                self.kwargs = {arg[0]: arg[1] for arg in self.kwargs}
            else:
                # Convert single arguments to dict
                assert isinstance(self.kwargs[0], str)
                tmp_kwargs = self.kwargs
                self.kwargs = {tmp_kwargs[0]: tmp_kwargs[1]}
        else:
            assert isinstance(self.kwargs, str), f"Unsupported kwargs type: {self.kwargs}"
            self.kwargs = {}

        if max_cycles is not None:
            self.kwargs["max_cycles"] = max_cycles
        if render_mode is not None:
            self.kwargs["render_mode"] = render_mode

    def common_observation_space(self):
        return self._common_observation_space

    def is_image(self):
        return self._is_image

    def get_agent_prefix(self):
        return self._agent_prefix

    def get_print_info(self):
        # Get print info from image encoder
        print_info = self._env.print_info
        self._env.print_info = None

        # Get print info from the current class
        if self.internal_print_info is not None:
            if print_info is None:
                print_info = self.internal_print_info
            else:
                print_info += self.internal_print_info
            self.internal_print_info = None

        # Return all print info
        return print_info

    def step(self, actions):
        """ Returns reward, terminated, info """

        # Fix the actions' type
        fixed_actions = {}
        for action_idx, action in enumerate(actions):
            if (
                    isinstance(action, (np.int64, np.int32, np.float64, np.float32)) or
                    (isinstance(action, np.ndarray) and str(action.dtype) in ["int64", "int32", "float64", "float32"])
            ):
                if len(action.flatten()) == 1:
                    tmp_action = action.item()
                else:
                    assert len(action.shape) == 1, f"len(action.shape): {len(action.shape)}"
                    tmp_action = action
            elif isinstance(action, (int, float)):
                tmp_action = action
            elif isinstance(action, list):
                tmp_action = np.array(action)
            elif isinstance(action, torch.Tensor):
                tmp_action = action.detach().cpu().item()
            else:
                raise NotImplementedError(f"Not supported action type! type(action): {type(action)}")
            fixed_actions[self._agent_prefix[action_idx]] = tmp_action

        # Apply action for each agent
        self._obs, rewards, terminations, truncations, self._info = self.step_function(fixed_actions)

        if self.sum_rewards is True:  # The case of fully cooperative tasks
            # Add all rewards together
            reward = float(sum(rewards.values()))
            # 'done' is True if there is at least a single truncation or termination
            done = (
                    any([termination for termination in terminations.values()]) or
                    any([truncation for truncation in truncations.values()])
            )
            # Keep only 'TimeLimit.truncated' in 'self._info'
            self._info = {'TimeLimit.truncated': self._info['TimeLimit.truncated']}
            return reward, done, {}

        else:  # The case of non-fully cooperative tasks
            # Handle the observation format. It should return a dictionary in the format of PettingZoo.
            if self._is_image is True:
                assert isinstance(self._obs, tuple)
                self._obs = {
                    _agent_prefix: self._obs[agent_id]
                    for agent_id, _agent_prefix in enumerate(self._agent_prefix)
                }
            else:
                assert isinstance(self._obs, dict)
            # Handle the timelimit truncation
            timelimit_truncated = self._info['TimeLimit.truncated']
            del self._info['TimeLimit.truncated']
            return (
                rewards,
                terminations,
                {'truncations': truncations, 'infos': self._info, 'TimeLimit.truncated': timelimit_truncated}
            )

    def get_obs(self):
        """ Returns all agent observations """
        return self._obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """

        # Image observations
        if self._is_image is True:
            assert (
                    (len(self.observation_space) == 1 and self.trainable_cnn is False) or
                    (len(self.observation_space) == 3 and self.trainable_cnn is True)
            ), f"'self.observation_space': {self.observation_space}, 'self.trainable_cnn': {self.trainable_cnn}"
            return self.observation_space[0] if self.trainable_cnn is False else self.observation_space

        # Vector observations
        else:
            if self._common_observation_space is False:
                return self.observation_space
            else:
                if (
                        (isinstance(self._obs, tuple) and len(self._obs) == 2) or
                        (isinstance(self._obs, dict) and len(self._obs) == self.n_agents)
                ):
                    if isinstance(self._obs, tuple):
                        assert all([self._obs[1][_agent_prefix] == {} for _agent_prefix in self._agent_prefix]), \
                            f"self._obs: {self._obs}"
                    return self.observation_space.shape
                else:
                    raise NotImplementedError

    def get_state(self):

        # Image observations
        if self._is_image is True:
            # The case of encoded images (both for env API and training framework)
            if self.trainable_cnn is False and self.centralized_image_encoding is False:
                if isinstance(self._obs, dict):  # The case of non-fully cooperative tasks
                    return np.concatenate([_obs for _obs in self._obs.values()], axis=0).astype(np.float32)
                else:  # The case of fully cooperative tasks
                    assert isinstance(self._obs, tuple)
                    return np.concatenate(self._obs, axis=0).astype(np.float32)
            # The case of raw images (both for env API and training framework)
            elif self.trainable_cnn is True and self.centralized_image_encoding is False:
                if isinstance(self._obs, dict):  # The case of non-fully cooperative tasks
                    return np.stack([_obs for _obs in self._obs.values()], axis=0).astype(np.float32)
                else:  # The case of fully cooperative tasks
                    assert isinstance(self._obs, tuple)
                    return np.stack(self._obs, axis=0).astype(np.float32)
            # The case of encoded images with centralized encoding (only for the training framework)
            elif self.trainable_cnn is False and self.centralized_image_encoding is True:
                # In this case, the centralized encoder will encode observations and combine them to create the state
                return None
            else:
                raise NotImplementedError()

        # Vector observations (only for non-fully cooperative tasks)
        else:
            if self._common_observation_space is False:
                assert isinstance(self._obs, dict)
                _obs = [self._obs[_agent_prefix] for _agent_prefix in self._agent_prefix]
                _obs = np.concatenate(_obs, axis=0).astype(np.float32)
                return _obs
            else:
                if (
                        (isinstance(self._obs, tuple) and len(self._obs) == 2) or
                        (isinstance(self._obs, dict) and len(self._obs) == self.n_agents)
                ):
                    if isinstance(self._obs, tuple):
                        assert all([self._obs[1][_agent_prefix] == {} for _agent_prefix in self._agent_prefix]), \
                            f"self._obs: {self._obs}"
                        self._obs = self._obs[0]
                    _obs = [self._obs[_agent_prefix] for _agent_prefix in self._agent_prefix]
                    return np.stack(_obs, axis=0).astype(np.float32)
                else:
                    raise NotImplementedError

    def get_state_size(self):
        """ Returns the shape of the state"""

        # Image observations
        if self._is_image is True:
            assert (
                    (len(self.observation_space) == 1 and self.trainable_cnn is False) or
                    (len(self.observation_space) == 3 and self.trainable_cnn is True)
            ), f"'self.observation_space': {self.observation_space}, 'self.trainable_cnn': {self.trainable_cnn}"
            return self.n_agents * self.observation_space[0] if self.trainable_cnn is False \
                                                             else \
                   (self.n_agents, *self.observation_space)

        # Vector observations
        else:
            if self._common_observation_space is False:
                raise NotImplementedError
            else:
                if (
                        (isinstance(self._obs, tuple) and len(self._obs) == 2) or
                        (isinstance(self._obs, dict) and len(self._obs) == self.n_agents)
                ):
                    if isinstance(self._obs, tuple):
                        assert all([self._obs[1][_agent_prefix] == {} for _agent_prefix in self._agent_prefix]), \
                            f"self._obs: {self._obs}"
                    return tuple((self.n_agents, *self.observation_space.shape))
                else:
                    raise NotImplementedError

    def get_avail_actions(self):

        if isinstance(self.action_space, int):
            avail_actions = []
            for agent_id in range(self.n_agents):
                avail_agent = self.get_avail_agent_actions(self._agent_prefix[agent_id])
                avail_actions.append(avail_agent)

            return avail_actions
        else:
            raise NotImplementedError

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        if isinstance(self.action_space, int):
            return self._env.action_space(agent_id).n * [1]  # 1 indicates availability of actions
        else:
            raise NotImplementedError

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        if isinstance(self.action_space, int):
            return self.action_space
        else:
            raise NotImplementedError

    def sample_actions(self):
        """ Returns a random sample of actions """
        sampled_actions = []
        for agent in self.original_env.agents:
            agent_sampled_action = self.original_env.action_space(agent).sample()
            if isinstance(agent_sampled_action, np.ndarray):
                assert self.key in ["waterworld_v4", "multiwalker_v9"], f"self.key: {self.key}"
                sampled_actions.append(agent_sampled_action)
            else:
                sampled_actions.append(int(agent_sampled_action))

        return sampled_actions

    def reset(self, seed=None):
        """ Returns initial observations and states"""

        # Control seed
        if seed is None:
            self._seed = self.np_random.choice(np.iinfo(np.int32).max)
        else:
            self.np_random, self._seed = seeding.np_random(self._seed)

        if self.key in ["entombed_cooperative_v3", "space_invaders_v2"]:

            # Here we fix the flickering issue of Atari 'entombed_cooperative_v3' and 'space_invaders_v2'
            # games when resetting the game.

            # Reset only the original environment and get the obs
            previous_observations, previous_infos = self.original_env.reset(seed=self._seed)
            previous_obs = list(previous_observations.values())[0]

            # Perform no action in order to sync obs and actions
            no_actions = {'first_0': 0, 'second_0': 0}
            observations, rewards, terminations, truncations, infos = self.original_env.step(no_actions)
            current_obs = list(observations.values())[1]
            reward = sum(rewards.values())
            assert reward == 0, f"Reward greater than 0 found during resetting the game! Reward: {reward}"

            # Get the first combined obs of agents
            if self.key == "entombed_cooperative_v3":
                obs = self._env.entombed_cooperative_v3_get_combined_images(previous_obs, current_obs)
            elif self.key == "space_invaders_v2":
                obs = self._env.space_invaders_v2_get_combined_images(previous_obs, current_obs)
            else:
                raise ValueError(f"self.key: {self.key}")

            # Preprocess obs
            self._obs = self._env.observation(obs)

            # Simulate TimeLimit wrapper
            self._env.env._elapsed_steps = 1

        else:
            self._obs, _ = self._env.reset(seed=self._seed)
            if self.sum_rewards is False:  # The case of non-fully cooperative tasks
                # Handle the observation format. It should return a dictionary in the format of PettingZoo.
                if self._is_image is True:
                    assert isinstance(self._obs, tuple)
                    self._obs = {
                        _agent_prefix: self._obs[agent_id]
                        for agent_id, _agent_prefix in enumerate(self._agent_prefix)
                    }
                else:
                    assert isinstance(self._obs, dict)

        return self.get_obs(), self.get_state()

    def render(self):
        if self.render_capable is True:
            try:
                if self.render_mode != "human":  # otherwise it is already rendered
                    # Get image
                    env_image = self.original_env.render()
                    # Convert RGB to BGR
                    env_image = cv2.cvtColor(env_image, cv2.COLOR_RGB2BGR)
                    # Render
                    cv2.imshow(f"Environment: {self.key}", env_image)
                    cv2.waitKey(1)
            except (Exception, SystemExit) as e:
                self.internal_print_info = (
                    "\n\n###########################################################"
                    f"\nError during rendering: \n\n{e}"
                    f"\n\nRendering will be disabled to continue the training."
                    "\n###########################################################\n"
                )
                self.render_capable = False

    def get_info(self):
        return self._info

    def get_n_agents(self):
        return self.n_agents

    def close(self):
        self._env.close()

    def seed(self):
        return self._seed

    def save_replay(self):
        pass

    @staticmethod
    def get_stats():
        return {}
