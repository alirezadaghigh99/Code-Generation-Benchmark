class AdaGrad(Optimizer):
    """The classic AdaGrad optimizer.

    See Also:
        - The functional AdaGrad optimizer: :func:`torchopt.adagrad`.
        - The differentiable meta-AdaGrad optimizer: :class:`torchopt.MetaAdaGrad`.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: ScalarOrSchedule = 1e-2,
        lr_decay: float = 0.0,
        weight_decay: float = 0.0,
        initial_accumulator_value: float = 0.0,
        eps: float = 1e-10,
        *,
        maximize: bool = False,
    ) -> None:
        r"""Initialize the AdaGrad optimizer.

        Args:
            params (iterable of Tensor): An iterable of :class:`torch.Tensor`\s. Specifies what
                tensors should be optimized.
            lr (float or callable, optional): This is a fixed global scaling factor or a learning
                rate scheduler. (default: :const:`1e-2`)
            lr_decay (float, optional): Learning rate decay. (default: :const:`0.0`)
            weight_decay (float, optional): Weight decay, add L2 penalty to parameters.
                (default: :const:`0.0`)
            initial_accumulator_value (float, optional): Initial value for the accumulator.
                (default: :const:`0.0`)
            eps (float, optional): A small constant applied to denominator outside of the square
                root (as in the Adam paper) to avoid dividing by zero when rescaling.
                (default: :const:`1e-10`)
            maximize (bool, optional): Maximize the params based on the objective, instead of
                minimizing. (default: :data:`False`)
        """
        super().__init__(
            params,
            alias.adagrad(
                lr=lr,
                lr_decay=lr_decay,
                weight_decay=weight_decay,
                initial_accumulator_value=initial_accumulator_value,
                eps=eps,
                maximize=maximize,
            ),
        )

