class CameraModel:
    r"""Class to represent camera models.

    Example:
        >>> # Pinhole Camera Model
        >>> cam = CameraModel(ImageSize(480, 640), CameraModelType.PINHOLE, torch.Tensor([328., 328., 320., 240.]))
        >>> # Brown Conrady Camera Model
        >>> cam = CameraModel(ImageSize(480, 640), CameraModelType.BROWN_CONRADY, torch.Tensor([1.0, 1.0, 1.0, 1.0,
        ... 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
        >>> # Kannala Brandt K3 Camera Model
        >>> cam = CameraModel(ImageSize(480, 640), CameraModelType.KANNALA_BRANDT_K3, torch.Tensor([1.0, 1.0, 1.0,
        ... 1.0, 1.0, 1.0, 1.0, 1.0]))
        >>> # Orthographic Camera Model
        >>> cam = CameraModel(ImageSize(480, 640), CameraModelType.ORTHOGRAPHIC, torch.Tensor([328., 328., 320., 240.]))
        >>> cam.params
        tensor([328., 328., 320., 240.])
    """

    def __init__(self, image_size: ImageSize, model_type: CameraModelType, params: Tensor) -> None:
        """Constructor method for CameraModel class.

        Args:
            image_size: Image size
            model_type: Camera model type
            params: Camera parameters of shape :math:`(B, N)`.
        """
        self._model = get_model_from_type(model_type, image_size, params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def __repr__(self) -> str:
        return f"CameraModel({self.image_size}, {self._model.__class__.__name__}, {self.params})"

