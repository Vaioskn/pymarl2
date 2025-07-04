from gymnasium import register

register(
    id=f'pressureplate-linear-4p-v0',
    entry_point='pymarlzooplus.envs.pressureplate_ai.pressureplate.environment:PressurePlate',
    kwargs={
        'height': 15,
        'width': 9,
        'n_agents': 4,
        'sensor_range': 4,
        'layout': 'linear'
    }
)

register(
    id=f'pressureplate-linear-5p-v0',
    entry_point='pymarlzooplus.envs.pressureplate_ai.pressureplate.environment:PressurePlate',
    kwargs={
        'height': 19,
        'width': 9,
        'n_agents': 5,
        'sensor_range': 4,
        'layout': 'linear'
    }
)

register(
    id=f'pressureplate-linear-6p-v0',
    entry_point='pymarlzooplus.envs.pressureplate_ai.pressureplate.environment:PressurePlate',
    kwargs={
        'height': 23,
        'width': 9,
        'n_agents': 6,
        'sensor_range': 4,
        'layout': 'linear'
    }
)

