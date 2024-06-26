def get_replay_buffer(policy, env, use_prioritized_rb=False,
                      use_nstep_rb=False, n_step=1, size=None):
    if policy is None or env is None:
        return None

    obs_shape = get_space_size(env.observation_space)
    kwargs = get_default_rb_dict(policy.memory_capacity, env)

    if size is not None:
        kwargs["size"] = size

    # on-policy policy
    if not issubclass(type(policy), OffPolicyAgent):
        kwargs["size"] = policy.horizon
        kwargs["env_dict"].pop("next_obs")
        kwargs["env_dict"].pop("rew")
        # TODO: Remove done. Currently cannot remove because of cpprb implementation
        # kwargs["env_dict"].pop("done")
        kwargs["env_dict"]["logp"] = {}
        kwargs["env_dict"]["ret"] = {}
        kwargs["env_dict"]["adv"] = {}
        if is_discrete(env.action_space):
            kwargs["env_dict"]["act"]["dtype"] = np.int32
        return ReplayBuffer(**kwargs)

    # N-step prioritized
    if use_prioritized_rb and use_nstep_rb:
        kwargs["Nstep"] = {"size": n_step,
                           "gamma": policy.discount,
                           "rew": "rew",
                           "next": "next_obs"}
        return PrioritizedReplayBuffer(**kwargs)

    if len(obs_shape) == 3:
        kwargs["env_dict"]["obs"]["dtype"] = np.ubyte
        kwargs["env_dict"]["next_obs"]["dtype"] = np.ubyte

    # prioritized
    if use_prioritized_rb:
        return PrioritizedReplayBuffer(**kwargs)

    # N-step
    if use_nstep_rb:
        kwargs["Nstep"] = {"size": n_step,
                           "gamma": policy.discount,
                           "rew": "rew",
                           "next": "next_obs"}
        return ReplayBuffer(**kwargs)

    return ReplayBuffer(**kwargs)