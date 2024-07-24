class CustomModel(NamedTuple):
    """
       A named tuple that can be used to initialize a custom PyTorch model.

       Args:
        name: The name of the custom model. Default is 'default_model'.
        model: The PyTorch model object which is a subclass of `torch.nn.Module` and implements the `forward` method and output a tensor of shape (batch_size x features). Alternatively, a __call__ method is also accepted.. Default is None.
        transform: A function that transforms a PIL.Image object into a PyTorch tensor that will be applied to each image before being fed to the model. Should correspond to the preprocessing logic of the supplied model. Default is None.
    """
    name: str = DEFAULT_MODEL_NAME
    model: Optional[torch.nn.Module] = None
    transform: Optional[Callable[[Image], torch.tensor]] = None

