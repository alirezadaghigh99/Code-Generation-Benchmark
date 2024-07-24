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

class Configuration:
    data_path: str = field(default="./", metadata={"help": "The input data directory."})
    batch_size: int = field(default=1, metadata={"help": "The number of batches for the training dataloader."})
    num_epochs: int = field(default=1, metadata={"help": "The number of epochs to run the training."})
    lr: float = field(default=1e-3, metadata={"help": "The learning rate to be used for the optimize."})
    output_path: str = field(default="./output", metadata={"help": "The output data directory."})
    image_size: Tuple[int, int] = field(default=(224, 224), metadata={"help": "The input image size."})

