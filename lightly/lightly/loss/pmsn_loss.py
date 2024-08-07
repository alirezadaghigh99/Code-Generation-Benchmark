class PMSNCustomLoss(MSNLoss):
    """Implementation of the loss function from PMSN [0] with a custom target
    distribution.

    - [0]: Prior Matching for Siamese Networks, 2022, https://arxiv.org/abs/2210.07277

    Attributes:
        target_distribution:
            A function that takes the mean anchor probabilities tensor with shape (dim,)
            as input and returns a target probability distribution tensor with the same
            shape. The returned distribution should sum up to one. The final
            regularization loss is calculated as KL(mean_anchor_probs, target_dist)
            where KL is the Kullback-Leibler divergence.
        temperature:
            Similarities between anchors and targets are scaled by the inverse of
            the temperature. Must be in (0, inf).
        sinkhorn_iterations:
            Number of sinkhorn normalization iterations on the targets.
        regularization_weight:
            Weight factor lambda by which the regularization loss is scaled. Set to 0
            to disable regularization.
        gather_distributed:
            If True, then target probabilities are gathered from all GPUs.

     Examples:

        >>> # define custom target distribution
        >>> def my_uniform_distribution(mean_anchor_probabilities: Tensor) -> Tensor:
        >>>     dim = mean_anchor_probabilities.shape[0]
        >>>     return mean_anchor_probabilities.new_ones(dim) / dim
        >>>
        >>> # initialize loss function
        >>> loss_fn = PMSNCustomLoss(target_distribution=my_uniform_distribution)
        >>>
        >>> # generate anchors and targets of images
        >>> anchors = transforms(images)
        >>> targets = transforms(images)
        >>>
        >>> # feed through PMSN model
        >>> anchors_out = model(anchors)
        >>> targets_out = model.target(targets)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(anchors_out, targets_out, prototypes=model.prototypes)
    """

    def __init__(
        self,
        target_distribution: Callable[[Tensor], Tensor],
        temperature: float = 0.1,
        sinkhorn_iterations: int = 3,
        regularization_weight: float = 1,
        gather_distributed: bool = False,
    ):
        super().__init__(
            temperature=temperature,
            sinkhorn_iterations=sinkhorn_iterations,
            regularization_weight=regularization_weight,
            gather_distributed=gather_distributed,
        )
        self.target_distribution = target_distribution

    def regularization_loss(self, mean_anchor_probs: Tensor) -> Tensor:
        """Calculates regularization loss with a custom target distribution."""
        target_dist = self.target_distribution(mean_anchor_probs).to(
            mean_anchor_probs.device
        )
        loss = F.kl_div(
            input=mean_anchor_probs.log(), target=target_dist, reduction="sum"
        )
        return loss

