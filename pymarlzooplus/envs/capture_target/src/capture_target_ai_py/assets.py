"""
Code adapted by:
https://github.com/yuchen-x/MacDeepMARL/blob/master/src/rlmamr/my_env/capture_target.py
"""

import numpy as np

NORTH = np.array([0, 1])
SOUTH = np.array([0, -1])
WEST = np.array([-1, 0])
EAST = np.array([1, 0])
STAY = np.array([0, 0])

TRANSLATION_TABLE = [
    # [left, intended_direction, right]
    [WEST,  NORTH, EAST],
    [EAST,  SOUTH, WEST],
    [SOUTH, WEST,  NORTH],
    [NORTH, EAST,  SOUTH],
    [STAY,  STAY,  STAY]
]

DIRECTION = np.array([[0.0, 1.0],
                      [0.0, -1.0],
                      [-1.0, 0.0],
                      [1.0, 0.0],
                      [0.0, 0.0]])

ACTIONS = ["NORTH", "SOUTH", "WEST", "EAST", "STAY"]