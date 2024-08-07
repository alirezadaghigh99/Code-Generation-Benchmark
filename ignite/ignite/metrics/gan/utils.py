class InceptionModel(torch.nn.Module):
    r"""Inception Model pre-trained on the ImageNet Dataset.

    Args:
        return_features: set it to `True` if you want the model to return features from the last pooling
            layer instead of prediction probabilities.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.
    """

    def __init__(self, return_features: bool, device: Union[str, torch.device] = "cpu") -> None:
        try:
            import torchvision
            from torchvision import models
        except ImportError:
            raise ModuleNotFoundError("This module requires torchvision to be installed.")
        super(InceptionModel, self).__init__()
        self._device = device
        if Version(torchvision.__version__) < Version("0.13.0"):
            model_kwargs = {"pretrained": True}
        else:
            model_kwargs = {"weights": models.Inception_V3_Weights.DEFAULT}

        self.model = models.inception_v3(**model_kwargs).to(self._device)

        if return_features:
            self.model.fc = torch.nn.Identity()
        else:
            self.model.fc = torch.nn.Sequential(self.model.fc, torch.nn.Softmax(dim=1))
        self.model.eval()

    @torch.no_grad()
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        if data.dim() != 4:
            raise ValueError(f"Inputs should be a tensor of dim 4, got {data.dim()}")
        if data.shape[1] != 3:
            raise ValueError(f"Inputs should be a tensor with 3 channels, got {data.shape}")
        if data.device != torch.device(self._device):
            data = data.to(self._device)
        return self.model(data)

