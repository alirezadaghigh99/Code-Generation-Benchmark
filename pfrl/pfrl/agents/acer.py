def compute_policy_gradient_loss(
    action,
    advantage,
    action_distrib,
    action_distrib_mu,
    action_value,
    v,
    truncation_threshold,
):
    """Compute policy gradient loss with off-policy bias correction."""
    assert np.isscalar(advantage)
    assert np.isscalar(v)
    log_prob = action_distrib.log_prob(action)
    if action_distrib_mu is not None:
        # Off-policy
        rho = compute_importance(action_distrib, action_distrib_mu, action)
        g_loss = 0
        if truncation_threshold is None:
            g_loss -= rho * log_prob * advantage
        else:
            # Truncated off-policy policy gradient term
            g_loss -= min(truncation_threshold, rho) * log_prob * advantage
            # Bias correction term
            if isinstance(action_distrib, torch.distributions.Categorical):
                g_loss += compute_policy_gradient_full_correction(
                    action_distrib=action_distrib,
                    action_distrib_mu=action_distrib_mu,
                    action_value=action_value,
                    v=v,
                    truncation_threshold=truncation_threshold,
                )
            else:
                g_loss += compute_policy_gradient_sample_correction(
                    action_distrib=action_distrib,
                    action_distrib_mu=action_distrib_mu,
                    action_value=action_value,
                    v=v,
                    truncation_threshold=truncation_threshold,
                )
    else:
        # On-policy
        g_loss = -log_prob * advantage
    return g_loss

