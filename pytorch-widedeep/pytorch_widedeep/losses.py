class RMSELoss(nn.Module):
    r"""Root mean square error loss adjusted for the possibility of using Label
    Smooth Distribution (LDS)

    LDS is based on
    [Delving into Deep Imbalanced Regression](https://arxiv.org/abs/2102.09554).
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, input: Tensor, target: Tensor, lds_weight: Optional[Tensor] = None
    ) -> Tensor:
        r"""
        Parameters
        ----------
        input: Tensor
            Input tensor with predictions (not probabilities)
        target: Tensor
            Target tensor with the actual classes
        lds_weight: Tensor, Optional
            Tensor of weights that will multiply the loss value.

        Examples
        --------
        >>> import torch
        >>> from pytorch_widedeep.losses import RMSELoss
        >>>
        >>> target = torch.tensor([1, 1.2, 0, 2]).view(-1, 1)
        >>> input = torch.tensor([0.6, 0.7, 0.3, 0.8]).view(-1, 1)
        >>> lds_weight = torch.tensor([0.1, 0.2, 0.3, 0.4]).view(-1, 1)
        >>> loss = RMSELoss()(input, target, lds_weight)
        """
        loss = (input - target) ** 2
        if lds_weight is not None:
            loss *= lds_weight
        return torch.sqrt(torch.mean(loss))

class MSLELoss(nn.Module):
    r"""Mean square log error loss with the option of using Label Smooth
    Distribution (LDS)

    LDS is based on
    [Delving into Deep Imbalanced Regression](https://arxiv.org/abs/2102.09554).
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, input: Tensor, target: Tensor, lds_weight: Optional[Tensor] = None
    ) -> Tensor:
        r"""
        Parameters
        ----------
        input: Tensor
            Input tensor with predictions (not probabilities)
        target: Tensor
            Target tensor with the actual classes
        lds_weight: Tensor, Optional
            Tensor of weights that will multiply the loss value.

        Examples
        --------
        >>> import torch
        >>> from pytorch_widedeep.losses import MSLELoss
        >>>
        >>> target = torch.tensor([1, 1.2, 0, 2]).view(-1, 1)
        >>> input = torch.tensor([0.6, 0.7, 0.3, 0.8]).view(-1, 1)
        >>> lds_weight = torch.tensor([0.1, 0.2, 0.3, 0.4]).view(-1, 1)
        >>> loss = MSLELoss()(input, target, lds_weight)
        """
        assert (
            input.min() >= 0
        ), """All input values must be >=0, if your model is predicting
            values <0 try to enforce positive values by activation function
            on last layer with `trainer.enforce_positive_output=True`"""
        assert target.min() >= 0, "All target values must be >=0"

        loss = (torch.log(input + 1) - torch.log(target + 1)) ** 2
        if lds_weight is not None:
            loss *= lds_weight
        return torch.mean(loss)

