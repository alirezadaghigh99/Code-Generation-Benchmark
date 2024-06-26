def compute_loss_with_kl_constraint(distrib, another_distrib, original_loss, delta):
    """Compute loss considering a KL constraint.

    Args:
        distrib (Distribution): Distribution to optimize
        another_distrib (Distribution): Distribution used to compute KL
        original_loss (torch.Tensor): Loss to minimize
        delta (float): Minimum KL difference
    Returns:
        torch.Tensor: new loss to minimize
    """
    distrib_params = get_params_of_distribution(distrib)
    for param in distrib_params:
        assert param.shape[0] == 1
        assert param.requires_grad
    # Compute g: a direction to minimize the original loss
    g = [
        grad[0]
        for grad in torch.autograd.grad(
            [original_loss], distrib_params, retain_graph=True
        )
    ]

    # Compute k: a direction to increase KL div.
    kl = torch.distributions.kl_divergence(another_distrib, distrib)
    k = [
        grad[0]
        for grad in torch.autograd.grad([-kl], distrib_params, retain_graph=True)
    ]

    # Compute z: combination of g and k to keep small KL div.
    kg_dot = sum(torch.dot(kp.flatten(), gp.flatten()) for kp, gp in zip(k, g))
    kk_dot = sum(torch.dot(kp.flatten(), kp.flatten()) for kp in k)
    if kk_dot > 0:
        k_factor = max(0, ((kg_dot - delta) / kk_dot))
    else:
        k_factor = 0
    z = [gp - k_factor * kp for kp, gp in zip(k, g)]
    loss = 0
    for p, zp in zip(distrib_params, z):
        loss += (p * zp).sum()
    return loss.reshape(original_loss.shape), float(kl)