class GPTAdamWFactory(OptimizerFactory):
    """AdamW optimizer for Decision Transformer architectures.

    .. code-block:: python

        from d3rlpy.optimizers import GPTAdamWFactory

        factory = GPTAdamWFactory(weight_decay=1e-4)

    Args:
        betas: coefficients used for computing running averages of
            gradient and its square.
        eps: term added to the denominator to improve numerical stability.
        weight_decay: weight decay (L2 penalty).
        amsgrad: flag to use the AMSGrad variant of this algorithm.
    """

    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0
    amsgrad: bool = False

    def create(
        self, named_modules: Iterable[Tuple[str, nn.Module]], lr: float
    ) -> AdamW:
        named_modules = list(named_modules)
        params_dict = {}
        decay = set()
        no_decay = set()
        for module_name, module in named_modules:
            for param_name, param in module.named_parameters():
                full_name = (
                    f"{module_name}.{param_name}" if module_name else param_name
                )

                if full_name not in params_dict:
                    params_dict[full_name] = param

                if param_name.endswith("bias"):
                    no_decay.add(full_name)
                elif param_name.endswith("weight") and isinstance(
                    module, (nn.Linear, nn.Conv2d)
                ):
                    decay.add(full_name)
                elif param_name.endswith("weight") and isinstance(
                    module, (nn.LayerNorm, nn.Embedding)
                ):
                    no_decay.add(full_name)

        # add non-catched parameters to no_decay
        all_names = set(params_dict.keys())
        remainings = all_names.difference(decay | no_decay)
        no_decay.update(remainings)
        assert len(decay | no_decay) == len(
            _get_parameters_from_named_modules(named_modules)
        )
        assert len(decay & no_decay) == 0

        optim_groups = [
            {
                "params": [params_dict[name] for name in sorted(list(decay))],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    params_dict[name] for name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        return AdamW(
            optim_groups,
            lr=lr,
            betas=self.betas,
            eps=self.eps,
            amsgrad=self.amsgrad,
        )

    @staticmethod
    def get_type() -> str:
        return "gpt_adam_w"

