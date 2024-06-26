def check_policy_on_policy_with_probe_env(
    env, algo_class, algo_args, learn_steps=5000, device="cpu"
):
    print(f"Probe environment: {type(env).__name__}")

    agent = algo_class(**algo_args, device=device)

    for _ in trange(learn_steps):
        state, _ = env.reset()
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []
        truncs = []

        for _ in range(100):
            action, log_prob, _, value = agent.get_action(np.expand_dims(state, 0))
            action = action[0]
            log_prob = log_prob[0]
            value = value[0]
            next_state, reward, done, trunc, _ = env.step(action)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            values.append(value)
            truncs.append(trunc)

            state = next_state
            if done:
                state, _ = env.reset()

        experiences = (
            states,
            actions,
            log_probs,
            rewards,
            dones,
            values,
            next_state,
        )
        agent.learn(experiences)

    for sample_obs, v_values in zip(env.sample_obs, env.v_values):
        state = torch.tensor(sample_obs).float().to(device)
        if v_values is not None:
            predicted_v_values = agent.critic(state).detach().cpu().numpy()[0]
            # print("---")
            # print("v", v_values, predicted_v_values)
            assert np.allclose(v_values, predicted_v_values, atol=0.1)

    if hasattr(env, "sample_actions"):
        for sample_action, policy_values in zip(env.sample_actions, env.policy_values):
            action = torch.tensor(sample_action).float().to(device)
            if policy_values is not None:
                predicted_policy_values = (
                    agent.actor(sample_obs).detach().cpu().numpy()[0]
                )
                # print("pol", policy_values, predicted_policy_values)
                assert np.allclose(policy_values, predicted_policy_values, atol=0.1)