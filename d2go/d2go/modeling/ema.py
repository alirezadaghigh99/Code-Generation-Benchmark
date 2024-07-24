def FromModel(cls, model: torch.nn.Module, device: str = "", **kwargs):
        ret = cls(**kwargs)
        ret.save_from(model, device)
        return ret

class EMAState(object):
    def __init__(self, include_frozen=True, include_buffer=True):
        self.include_frozen = include_frozen
        self.include_buffer = include_buffer
        self.state = {}
        # HACK: This hack is needed to strip checkpoint wrapper prefix from fqns so it doesn't affect loading.
        # TODO: Remove this hack by rewriting EMAState to use model.state_dict()
        self.prefix_to_remove = [_CHECKPOINT_PREFIX]

    @classmethod
    def FromModel(cls, model: torch.nn.Module, device: str = "", **kwargs):
        ret = cls(**kwargs)
        ret.save_from(model, device)
        return ret

    def save_from(self, model: torch.nn.Module, device: str = ""):
        """Save model state from `model` to this object"""
        for name, val in self.get_model_state_iterator(model):
            val = val.detach().clone()
            self.state[name] = val.to(device) if device else val

    def apply_to(self, model: torch.nn.Module):
        """Apply state to `model` from this object"""
        with torch.no_grad():
            for name, val in self.get_model_state_iterator(model):
                assert (
                    name in self.state
                ), f"Name {name} not existed, available names {self.state.keys()}"
                val.copy_(self.state[name])

    @contextmanager
    def apply_and_restore(self, model):
        old_state = EMAState.FromModel(model, self.device)
        self.apply_to(model)
        yield old_state
        old_state.apply_to(model)

    def get_ema_model(self, model):
        ret = copy.deepcopy(model)
        self.apply_to(ret)
        return ret

    @property
    def device(self):
        if not self.has_inited():
            return None
        return next(iter(self.state.values())).device

    def to(self, device):
        for name in self.state:
            self.state[name] = self.state[name].to(device)
        return self

    def has_inited(self):
        return self.state

    def clear(self):
        self.state.clear()
        return self

    def _get_model_parameter_iterator(self, model):
        """
        Return iterator for model parameters. Remove frozen parameters if needed.
        """
        for name, params in model.named_parameters():
            if params.requires_grad or self.include_frozen:
                yield name, params

    def get_model_state_iterator(self, model):
        param_iter = self._get_model_parameter_iterator(model)
        if self.include_buffer:
            param_iter = itertools.chain(param_iter, model.named_buffers())
        return _remove_prefix(param_iter, self.prefix_to_remove)

    def state_dict(self):
        return self.state

    def load_state_dict(self, state_dict, strict: bool = True):
        self.clear()
        for x, y in state_dict.items():
            self.state[x] = y
        return torch.nn.modules.module._IncompatibleKeys(
            missing_keys=[], unexpected_keys=[]
        )

    def __repr__(self):
        ret = f"EMAState(state=[{','.join(self.state.keys())}])"
        return ret

class EMAUpdater(object):
    """Model Exponential Moving Average
    Keep a moving average of everything in the model state_dict (parameters and
    buffers). This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    Note:  It's very important to set EMA for ALL network parameters (instead of
    parameters that require gradient), including batch-norm moving average mean
    and variance.  This leads to significant improvement in accuracy.
    For example, for EfficientNetB3, with default setting (no mixup, lr exponential
    decay) without bn_sync, the EMA accuracy with EMA on params that requires
    gradient is 79.87%, while the corresponding accuracy with EMA on all params
    is 80.61%.

    Also, bn sync should be switched on for EMA.
    """

    def __init__(
        self,
        state: EMAState,
        decay: float = 0.999,
        device: str = "",
        use_lerp: bool = False,
        decay_warm_up_factor: int = -1,
    ):
        self.decay = decay
        self.device = device

        self.state = state
        self.use_lerp = use_lerp
        self.debug_lerp = False

        self._num_updates: int = -1
        self.decay_warm_up_factor = decay_warm_up_factor
        if self.decay_warm_up_factor >= 0:
            self._num_updates = 0

    def init_state(self, model):
        self.state.clear()
        self.state.save_from(model, self.device)

    def update(self, model):
        # compute decay
        decay = self.decay
        if self._num_updates >= 0:
            self._num_updates += 1
            decay = min(
                self.decay,
                (1 + self._num_updates)
                / (self.decay_warm_up_factor + self._num_updates),
            )

        # update moving average
        with torch.no_grad():
            ema_param_list = []
            param_list = []
            for name, val in self.state.get_model_state_iterator(model):
                ema_val = self.state.state[name]
                if self.device:
                    val = val.to(self.device)
                if val.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                    ema_param_list.append(ema_val)
                    param_list.append(val)
                else:
                    ema_val.copy_(ema_val * decay + val * (1.0 - decay))
            self._ema_avg(ema_param_list, param_list, decay)

    def _ema_avg(
        self,
        averaged_model_parameters: List[torch.Tensor],
        model_parameters: List[torch.Tensor],
        decay: float,
    ) -> None:
        """
        Function to perform exponential moving average:
        x_avg = alpha * x_avg + (1-alpha)* x_t
        """
        if self.use_lerp:
            if self.debug_lerp:
                orig_averaged_model_parameters = torch._foreach_mul(
                    averaged_model_parameters, decay
                )
                torch._foreach_add_(
                    orig_averaged_model_parameters, model_parameters, alpha=1 - decay
                )

            torch._foreach_lerp_(
                averaged_model_parameters, model_parameters, 1.0 - decay
            )
            if self.debug_lerp:
                for orig_val, lerp_val in zip(
                    orig_averaged_model_parameters, averaged_model_parameters
                ):
                    assert torch.allclose(orig_val, lerp_val, rtol=1e-4, atol=1e-3)
        else:
            torch._foreach_mul_(averaged_model_parameters, decay)
            torch._foreach_add_(
                averaged_model_parameters, model_parameters, alpha=1 - decay
            )

