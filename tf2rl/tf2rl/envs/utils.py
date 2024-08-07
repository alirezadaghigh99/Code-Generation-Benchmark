def make(id, **kwargs):
    r"""
    Make gym.Env with version tolerance

    Args:
        id (str) : Id specifying `gym.Env` registered to `gym.env.registry`.
                   Valid format is `"^(?:[\w:-]+\/)?([\w:.-]+)-v(\d+)$"`
                   See https://github.com/openai/gym/blob/v0.21.0/gym/envs/registration.py#L17-L19

    Returns:
        gym.Env : Environment
    """
    try:
        return gym.make(id, **kwargs)
    except gym.error.DeprecatedEnv:
        # `gym.make` (v0.21.0) check only version mismatch for `DeprecatedEnv`.
        # https://github.com/openai/gym/blob/v0.21.0/gym/envs/registration.py#L162-L189
        logger.warning(f"Version Mismatch: {id}")
        env_idv = id.rsplit("-v", 1)

        candidate = [e for e in gym.envs.registry.env_specs.keys()
                     if env_idv[0] == e.rsplit("-v", 1)[0]]
        if len(candidate) == 0:
            raise

        new_v = max(map(lambda _id: int(_id.rsplit("-v", 1)[1]), candidate))
        id = f"{env_idv[0]}-v{new_v}"
        logger.warning(f"Use {id}")
        return gym.make(id, **kwargs)

