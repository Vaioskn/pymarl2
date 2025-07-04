from gymnasium import register

import pymarlzooplus.envs.multiagent_particle_envs.mpe.scenarios as scenarios

simple_spread_scenarios = {
    "simple_spread_3": "SimpleSpread-3-v0",
    "simple_spread_4": "SimpleSpread-4-v0",
    "simple_spread_5": "SimpleSpread-5-v0",
    "simple_spread_8": "SimpleSpread-8-v0",
}

for scenario_name, gymkey in simple_spread_scenarios.items():

    # Get the number of agents and landmarks and then remove it from the 'scenario_name'
    n_agents_landmarks = int(scenario_name.split('_')[2])
    scenario_name = scenario_name.replace(f"_{n_agents_landmarks}", "")

    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world(n_agents_landmarks=n_agents_landmarks)

    # Registers multi-agent particle environments:
    register(
        gymkey,
        entry_point="pymarlzooplus.envs.multiagent_particle_envs.mpe.environment:MultiAgentEnv",
        kwargs={
            "world": world,
            "reset_callback": scenario.reset_world,
            "reward_callback": scenario.reward,
            "observation_callback": scenario.observation,
        },
    )
