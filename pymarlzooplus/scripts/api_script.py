# Import packages
from pymarlzooplus.envs import REGISTRY as env_REGISTRY

##################################### API for fully cooperative tasks. ###################################
## 'n_agns' (int) is the number of agents in the environment.
## 'n_acts' (int) is the number of actions available to each agent (all the agents have the same actions).
## 'reward' (float) is the sum of all agents' rewards.
## 'done' (bool) is False if at least an agent's done is False.
## 'extra_info' (dict) typically is an empty dictionary.
## 'info' (dict) contains only 'TimeLimit.truncated' (bool) which is False if at least an agent's truncated is False.
## 'obs' (tuple) contains numpy arrays, each of which corresponds to an agent:
##               a) In the case of encoding the images, the shape of each observation is (cnn_features_dim,),
##                  as defined by the argument 'cnn_features_dim' (default is 128).
##               b) In the case of raw images, the shape of each observation is (3, h, w).
## 'state' (np.ndarray) is the concatenation of all observations:
##                      a) In the case of encoding the images, the shape is (cnn_features_dim * n_agns,).
##                      b) In the case of raw images, the shape is (n_agns, 3, h, w).

# # Example of arguments for PettingZoo.
# # Specifically:
#   - Butterfly (except from "Knights Archers Zombies"),
#   - Atari (only "Emtombed: Cooperative" and "Space Invaders")

args = {
  "env": "pettingzoo",
  "env_args": {
      "key": "pistonball_v6",
      "time_limit": 900,  # Episode horizon.
      "render_mode": "rgb_array",  # Options: "human", "rgb_array
      "image_encoder": "ResNet18",  # Options: "ResNet18", "SlimSAM", "CLIP"
      "image_encoder_use_cuda": True,  # Whether to load image-encoder in GPU or not.
      "image_encoder_batch_size": 10,  # How many images to encode in a single pass.
      "partial_observation": False,  # Only for "Emtombed: Cooperative" and "Space Invaders"
      "trainable_cnn": False,  # Specifies whether to return image-observation or the encoded vector-observation
      "kwargs": "",
      "seed": 2024
  }
}

# # Example of arguments for Overcooked
# args = {
#   "env": "overcooked",
#   "env_args": {
#       "key": "coordination_ring",
#       "time_limit": 500,
#       "reward_type": "sparse",
#       "seed": 2024
#   }
# }

# # Example of arguments for Pressure Plate
# args = {
#   "env": "pressureplate",
#   "env_args": {
#       "key": "pressureplate-linear-4p-v0",
#       "time_limit": 500,
#       "seed": 2024
#   }
# }

# # Example of arguments for LBF version 2
# args = {
#   "env": "gymma",
#   "env_args": {
#       "key": "lbforaging:Foraging-4s-11x11-3p-2f-coop-v2",
#       "time_limit": 50,
#       "seed": 2024
#   }
# }

# # Example of arguments for LBF version 3
# args = {
#   "env": "gymma",
#   "env_args": {
#       "key": "lbforaging:Foraging-4s-11x11-3p-2f-coop-v3",
#       "time_limit": 50,
#       "seed": 2024
#   }
# }

# # Example of arguments for RWARE version 1
# args = {
#   "env": "gymma",
#   "env_args": {
#       "key": "rware:rware-small-4ag-hard-v1",
#       "time_limit": 500,
#       "seed": 2024
#   }
# }

# # Example of arguments for RWARE version 2
# args = {
#   "env": "gymma",
#   "env_args": {
#       "key": "rware:rware-small-4ag-hard-v2",
#       "time_limit": 500,
#       "seed": 2024
#   }
# }

# # Example of arguments for MPE
# args = {
#   "env": "gymma",
#   "env_args": {
#       "key": "mpe:SimpleSpeakerListener-v0",
#       "time_limit": 25,
#       "seed": 2024
#   }
# }

# # Example of arguments for Capture Target
# args = {
#   "env": "capturetarget",
#   "env_args": {
#       "key": "CaptureTarget-6x6-1t-2a-v0",
#       "time_limit": 60,
#       "seed": 2024
#   }
# }

# # Example of arguments for Box Pushing
# args = {
#   "env": "boxpushing",
#   "env_args": {
#       "key": "BoxPushing-6x6-2a-v0",
#       "time_limit": 60,
#       "seed": 2024
#   }
# }

# Initialize environment
env = env_REGISTRY[args["env"]](**args["env_args"])
n_agns = env.get_n_agents()
n_acts = env.get_total_actions()
# Reset the environment
obs, state = env.reset()
done = False
# Run an episode
while not done:
    # Render the environment (optional)
    env.render()
    # Insert the policy's actions here
    actions = env.sample_actions()
    # Apply an environment step
    reward, done, extra_info = env.step(actions)
    obs = env.get_obs()
    state = env.get_state()
    info = env.get_info()
# Terminate the environment
env.close()

##########################################################################################################


####################################### API for NON-fully cooperative tasks, two cases: ##############################
## 'n_agns' (int) is the number of agents in the environment.
## 'common_observation_space' (bool) whether the observations is the shame for all agents
## - 'reward' (dict) is returned as provided by PettingZoo,
##                   where each element is either np.int64, int, np.float64, or float.
## - 'done' (dict) is returned as provided by PettingZoo, where each element is bool.
## - 'info' (dict) contains 'TimeLimit.truncated' (bool) which is False if at least an agent's truncated is False,
##                 as well as 'truncations' and 'infos' dictionaries as returned by PettingZoo.
## - 'obs' (dict) is returned in the same format as given by PettingZoo (i.e, the same keys), but in case of images:
##                a) If images are encoded (i.e., 'trainable_cnn' is True), the shape of each
##                   observation (np.ndarray) is (cnn_features_dim,) as defined by the argument
##                  'cnn_features_dim' (default is 128).
##                b) If raw images are used (i.e., 'trainable_cnn' is False), the shape of each
##                   observation (np.ndarray) is (3, h, w).
## - 'state' (np.ndarray) is the concatenation of all observations:
##                        a) In the case of encoding the images, the shape is (cnn_features_dim * n_agns,).
##                        b) In the case of raw images, the shape is (n_agns, 3, h, w).
##                        c) In the case of vector observations, the shape is (obs_dim * n_agns,).

# # Arguments for PettingZoo.
# # Specifically:
# # - Atari (except from "Emtombed: Cooperative" and "Space Invaders"),
# # - Butterfly (only "Knights Archers Zombies"),
# # - MPE, and SISL

args = {
  "env": "pettingzoo",
  "env_args": {
      "key": "basketball_pong_v3",
      "time_limit": 900,
      "render_mode": "rgb_array",
      "image_encoder": "ResNet18",
      "image_encoder_use_cuda": False,
      "image_encoder_batch_size": 10,
      "trainable_cnn": False,
      "kwargs": "",
      "seed": 2024
  }
}

# Initialize environment
env = env_REGISTRY[args["env"]](**args["env_args"])
n_agns = env.get_n_agents()
common_observation_space = env.common_observation_space()
# Reset the environment
obs, state = env.reset()
done = False
# Run an episode
while not done:
    # Render the environment (optional)
    env.render()
    # Insert the policy's actions here
    actions = env.sample_actions()
    # Apply an environment step
    reward, done, info = env.step(actions)
    done = all([agent_done for agent_done in done.values()])
    obs = env.get_obs()
    state = env.get_state()
# Terminate the environment
env.close()

##########################################################################################################


####################################### API for Classic environment ###################################
## We return the original PettingZoo environment, where:
## 'observation':
##      a) All except 'rps_v2': (dict) contains 'observation' (np.array) and 'action_mask' (np.array)
##      b) 'rps_v2': (np.array)
## 'reward' (np.int64 | int | np.float64 | float)
## 'termination' (bool)
## 'truncation' (bool)
## 'info' (dict):
##      a) All except 'hanabi_v5': empty
##      b) 'hanabi_v5': contains 'action_mask' (np.ndarray) which is equal to observation['action_mask']

# # Arguments for PettingZoo Classic
# args = {
#   "env": "pettingzoo",
#   "env_args": {
#       "key": "chess_v6",
#       "render_mode": "human",
#       "kwargs": "",
#   }
# }

# # Initialize environment
# env = env_REGISTRY[args["env"]](**args["env_args"]).original_env
# # Reset environment
# env.reset(seed=42)
# # Run an episode
# for agent in env.agent_iter():
#     # Render the environment (optional)
#     env.render()
#     # Get environment data
#     observation, reward, termination, truncation, info = env.last()
#     # Get action
#     if termination or truncation:
#         action = None
#     else:
#         mask = observation["action_mask"]  # For 'Rock Paper Scissors', comment out this line
#         # this is where you would insert your policy
#         action = env.action_space(agent).sample(mask)  # For 'Rock Paper Scissors', don't use the 'mask'
#     # Apply an environment step
#     env.step(action)
# env.close()

##########################################################################################################
