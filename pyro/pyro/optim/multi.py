class MixedMultiOptimizer(MultiOptimizer):
    """
    Container class to combine different :class:`MultiOptimizer` instances for
    different parameters.

    :param list parts: A list of ``(names, optim)`` pairs, where each
        ``names`` is a list of parameter names, and each ``optim`` is a
        :class:`MultiOptimizer` or :class:`~pyro.optim.optim.PyroOptim` object
        to be used for the named parameters. Together the ``names`` should
        partition up all desired parameters to optimize.
    :raises ValueError: if any name is optimized by multiple optimizers.
    """

    def __init__(self, parts: List) -> None:
        optim_dict: Dict = {}
        self.parts = []
        for names_part, optim in parts:
            if isinstance(optim, PyroOptim):
                optim = PyroMultiOptimizer(optim)
            for name in names_part:
                if name in optim_dict:
                    raise ValueError(
                        "Attempted to optimize parameter '{}' by two different optimizers: "
                        "{} vs {}".format(name, optim_dict[name], optim)
                    )
                optim_dict[name] = optim
            self.parts.append((names_part, optim))

    def step(self, loss: torch.Tensor, params: Dict):
        for names_part, optim in self.parts:
            optim.step(loss, {name: params[name] for name in names_part})

    def get_step(self, loss: torch.Tensor, params: Dict) -> Dict:
        updated_values = {}
        for names_part, optim in self.parts:
            updated_values.update(
                optim.get_step(loss, {name: params[name] for name in names_part})
            )
        return updated_values

