class SwaVLoss(nn.Module):
    """Implementation of the SwaV loss.

    Attributes:
        temperature:
            Temperature parameter used for cross entropy calculations.
        sinkhorn_iterations:
            Number of iterations of the sinkhorn algorithm.
        sinkhorn_epsilon:
            Temperature parameter used in the sinkhorn algorithm.
        sinkhorn_gather_distributed:
            If True then features from all gpus are gathered to calculate the
            soft codes in the sinkhorn algorithm.

    """

    def __init__(
        self,
        temperature: float = 0.1,
        sinkhorn_iterations: int = 3,
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_gather_distributed: bool = False,
    ):
        super(SwaVLoss, self).__init__()
        if sinkhorn_gather_distributed and not dist.is_available():
            raise ValueError(
                "sinkhorn_gather_distributed is True but torch.distributed is not "
                "available. Please set gather_distributed=False or install a torch "
                "version with distributed support."
            )

        self.temperature = temperature
        self.sinkhorn_iterations = sinkhorn_iterations
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_gather_distributed = sinkhorn_gather_distributed

    def subloss(self, z: torch.Tensor, q: torch.Tensor):
        """Calculates the cross entropy for the SwaV prediction problem.

        Args:
            z:
                Similarity of the features and the SwaV prototypes.
            q:
                Codes obtained from Sinkhorn iterations.

        Returns:
            Cross entropy between predictions z and codes q.

        """
        return -torch.mean(
            torch.sum(q * F.log_softmax(z / self.temperature, dim=1), dim=1)
        )

    def forward(
        self,
        high_resolution_outputs: List[torch.Tensor],
        low_resolution_outputs: List[torch.Tensor],
        queue_outputs: List[torch.Tensor] = None,
    ):
        """Computes the SwaV loss for a set of high and low resolution outputs.

        Args:
            high_resolution_outputs:
                List of similarities of features and SwaV prototypes for the
                high resolution crops.
            low_resolution_outputs:
                List of similarities of features and SwaV prototypes for the
                low resolution crops.
            queue_outputs:
                List of similarities of features and SwaV prototypes for the
                queue of high resolution crops from previous batches.

        Returns:
            Swapping assignments between views loss (SwaV) as described in [0].

        [0]: SwaV, 2020, https://arxiv.org/abs/2006.09882

        """
        n_crops = len(high_resolution_outputs) + len(low_resolution_outputs)

        # multi-crop iterations
        loss = 0.0
        for i in range(len(high_resolution_outputs)):
            # compute codes of i-th high resolution crop
            with torch.no_grad():
                outputs = high_resolution_outputs[i].detach()

                # Append queue outputs
                if queue_outputs is not None:
                    outputs = torch.cat((outputs, queue_outputs[i].detach()))

                # Compute the codes
                q = sinkhorn(
                    outputs,
                    iterations=self.sinkhorn_iterations,
                    epsilon=self.sinkhorn_epsilon,
                    gather_distributed=self.sinkhorn_gather_distributed,
                )

                # Drop queue similarities
                if queue_outputs is not None:
                    q = q[: len(high_resolution_outputs[i])]

            # compute subloss for each pair of crops
            subloss = 0.0
            for v in range(len(high_resolution_outputs)):
                if v != i:
                    subloss += self.subloss(high_resolution_outputs[v], q)

            for v in range(len(low_resolution_outputs)):
                subloss += self.subloss(low_resolution_outputs[v], q)

            loss += subloss / (n_crops - 1)

        return loss / len(high_resolution_outputs)

