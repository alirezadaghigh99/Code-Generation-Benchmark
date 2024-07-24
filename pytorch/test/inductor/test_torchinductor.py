def repeat_interleave(x, n):
            # e.g. x=[1, 2, 3], n=2 => returns [1, 1, 2, 2, 3, 3]
            i = torch.arange(x.shape[0] * n, device=x.device)
            return x[i // n]

def copy(x):
            i = torch.arange(x.size(0), device=x.device)
            return x[i]

class BatchNorm(torch.nn.BatchNorm2d):
            def __init__(
                self,
                num_features,
                eps=1e-5,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
                device=None,
                dtype=None,
            ):
                factory_kwargs = {"device": device, "dtype": dtype}
                super().__init__(
                    num_features,
                    eps=eps,
                    momentum=momentum,
                    affine=affine,
                    track_running_stats=track_running_stats,
                    **factory_kwargs,
                )

            def forward(self, x):
                if self.momentum is None:
                    exponential_average_factor = 0.0
                else:
                    exponential_average_factor = self.momentum

                if self.training and self.track_running_stats:
                    # TODO: if statement only here to tell the jit to skip emitting this when it is None
                    if self.num_batches_tracked is not None:  # type: ignore[has-type]
                        self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
                        if self.momentum is None:  # use cumulative moving average
                            exponential_average_factor = 1.0 / float(
                                self.num_batches_tracked
                            )
                        else:  # use exponential moving average
                            exponential_average_factor = self.momentum
                if self.training:
                    bn_training = True
                else:
                    bn_training = (self.running_mean is None) and (
                        self.running_var is None
                    )
                x = F.batch_norm(
                    x,
                    # If buffers are not to be tracked, ensure that they won't be updated
                    (
                        self.running_mean
                        if not self.training or self.track_running_stats
                        else None
                    ),
                    (
                        self.running_var
                        if not self.training or self.track_running_stats
                        else None
                    ),
                    self.weight,
                    self.bias,
                    bn_training,
                    exponential_average_factor,
                    self.eps,
                )
                return x

