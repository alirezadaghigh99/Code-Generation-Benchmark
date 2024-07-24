class RBSparsifyingWeight(BinaryMask, StatefullModuleInterface):
    WEIGHTS_SHAPE_KEY = "weight_shape"
    FROZEN_KEY = "frozen"
    COMPRESSION_LR_MULTIPLIER_KEY = "compression_lr_multiplier"
    EPS_KEY = "eps"

    def __init__(self, weight_shape: List[int], frozen=True, compression_lr_multiplier=None, eps=1e-6):
        super().__init__(weight_shape)
        self.frozen = frozen
        self.eps = eps
        self._mask = CompressionParameter(
            logit(torch.ones(weight_shape) * 0.99),
            requires_grad=not self.frozen,
            compression_lr_multiplier=compression_lr_multiplier,
        )
        self._compression_lr_multiplier = compression_lr_multiplier
        self.binary_mask = binary_mask(self._mask)
        self.register_buffer("uniform", torch.zeros(weight_shape))
        self.mask_calculation_hook = MaskCalculationHook(self)

    @property
    def mask(self) -> torch.nn.Parameter:
        return self._mask

    @mask.setter
    def mask(self, tensor: torch.Tensor):
        self._mask.data = tensor
        self.binary_mask = binary_mask(self._mask)

    def _calc_training_binary_mask(self, weight):
        u = self.uniform if self.training and not self.frozen else None
        return calc_rb_binary_mask(self._mask, u, self.eps)

    def loss(self):
        return binary_mask(self._mask)

    def get_config(self) -> Dict[str, Any]:
        return {
            self.WEIGHTS_SHAPE_KEY: list(self.mask.shape),
            self.FROZEN_KEY: self.frozen,
            self.COMPRESSION_LR_MULTIPLIER_KEY: self._compression_lr_multiplier,
            self.EPS_KEY: self.eps,
        }

    @classmethod
    def from_config(cls, state: Dict[str, Any]) -> "RBSparsifyingWeight":
        return RBSparsifyingWeight(
            weight_shape=state[cls.WEIGHTS_SHAPE_KEY],
            frozen=state[cls.FROZEN_KEY],
            compression_lr_multiplier=state[cls.COMPRESSION_LR_MULTIPLIER_KEY],
            eps=state[cls.EPS_KEY],
        )

