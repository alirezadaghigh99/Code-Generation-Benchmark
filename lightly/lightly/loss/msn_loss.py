def sinkhorn(
    probabilities: Tensor,
    iterations: int = 3,
    gather_distributed: bool = False,
) -> Tensor:
    """Runs sinkhorn normalization on the probabilities as described in [0].

    Code inspired by [1].

    - [0]: Masked Siamese Networks, 2022, https://arxiv.org/abs/2204.07141
    - [1]: https://github.com/facebookresearch/msn

    Args:
        probabilities:
            Probabilities tensor with shape (batch_size, num_prototypes).
        iterations:
            Number of iterations of the sinkhorn algorithms. Set to 0 to disable.
        gather_distributed:
            If True then features from all gpus are gathered during normalization.
    Returns:
        A normalized probabilities tensor.

    """
    if iterations <= 0:
        return probabilities

    world_size = 1
    if gather_distributed and dist.is_initialized():
        world_size = dist.get_world_size()

    num_targets, num_prototypes = probabilities.shape
    probabilities = probabilities.T
    sum_probabilities = torch.sum(probabilities)
    if world_size > 1:
        dist.all_reduce(sum_probabilities)
    probabilities = probabilities / sum_probabilities

    for _ in range(iterations):
        # normalize rows
        row_sum = torch.sum(probabilities, dim=1, keepdim=True)
        if world_size > 1:
            dist.all_reduce(row_sum)
        probabilities /= row_sum
        probabilities /= num_prototypes

        # normalize columns
        probabilities /= torch.sum(probabilities, dim=0, keepdim=True)
        probabilities /= num_targets

    probabilities *= num_targets
    return probabilities.T

