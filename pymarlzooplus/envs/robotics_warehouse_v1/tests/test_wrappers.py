import pytest

from pymarlzooplus.envs.robotics_warehouse_v1.rware_v1.warehouse import Warehouse, RewardType


@pytest.fixture
def env_single_agent():
    env = Warehouse(3, 8, 3, 1, 0, 1, 5, None, None, RewardType.GLOBAL)
    env.reset()
    return env


@pytest.fixture
def env_double_agent():
    env = Warehouse(3, 8, 3, 2, 0, 1, 5, None, None, RewardType.GLOBAL)
    env.reset()
    return env


@pytest.fixture
def env_double_agent_with_msg():
    env = Warehouse(3, 8, 3, 2, 2, 1, 5, None, None, RewardType.GLOBAL)
    env.reset()
    return env

