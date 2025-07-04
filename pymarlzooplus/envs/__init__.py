# Needed for the imports
REGISTRY_availability = [
    "gymma",
    "pettingzoo",
    "overcooked",
    "pressureplate",
    "capturetarget",
    "boxpushing",
]

from functools import partial  # noqa: E402

from pymarlzooplus.envs.multiagentenv import MultiAgentEnv  # noqa: E402
from pymarlzooplus.envs.gym_wrapper import _GymmaWrapper  # noqa: E402
from pymarlzooplus.envs.pettingzoo_wrapper import _PettingZooWrapper  # noqa: E402
from pymarlzooplus.envs.overcooked_wrapper import _OvercookedWrapper  # noqa: E402
from pymarlzooplus.envs.pressureplate_wrapper import _PressurePlateWrapper  # noqa: E402
from pymarlzooplus.envs.capturetarget_wrapper import _CaptureTargetWrapper  # noqa: E402
from pymarlzooplus.envs.boxpushing_wrapper import _BoxPushingWrapper  # noqa: E402

# Gymnasium registrations
import pymarlzooplus.envs.lbf_registration_v2  # noqa: E402
import pymarlzooplus.envs.lbf_registration  # noqa: E402
import pymarlzooplus.envs.mpe_registration  # noqa: E402
import pymarlzooplus.envs.rware_v1_registration  # noqa: E402


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {
    "gymma": partial(env_fn, env=_GymmaWrapper),
    "pettingzoo": partial(env_fn, env=_PettingZooWrapper),
    "overcooked": partial(env_fn, env=_OvercookedWrapper),
    "pressureplate": partial(env_fn, env=_PressurePlateWrapper),
    "capturetarget": partial(env_fn, env=_CaptureTargetWrapper),
    "boxpushing": partial(env_fn, env=_BoxPushingWrapper)
}


