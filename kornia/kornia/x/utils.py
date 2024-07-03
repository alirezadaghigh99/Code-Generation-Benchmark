class Lambda(Module):
    """Module to create a lambda function as Module.

    Args:
        fcn: a pointer to any function.

    Example:
        >>> import torch
        >>> import kornia as K
        >>> fcn = Lambda(lambda x: K.geometry.resize(x, (32, 16)))
        >>> fcn(torch.rand(1, 4, 64, 32)).shape
        torch.Size([1, 4, 32, 16])
    """

    def __init__(self, fcn: Callable[..., Any]) -> None:
        super().__init__()
        self.fcn = fcn

    def forward(self, x: Tensor) -> Any:
        return self.fcn(x)