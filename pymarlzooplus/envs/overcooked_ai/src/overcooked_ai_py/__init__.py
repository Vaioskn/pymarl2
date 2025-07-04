from gymnasium import register

register(
    id="Overcooked-v0",
    entry_point="pymarlzooplus.envs.overcooked_ai.src.overcooked_ai_py.mdp.overcooked_env:Overcooked",
)
