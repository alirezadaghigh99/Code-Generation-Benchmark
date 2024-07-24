class BYOL(nn.Module):
    """BYOL implementation.

    BYOL contains two identical backbone networks. The first is trained as usual, and
    its weights are updated with each training batch. The second, "target" network,
    is updated using a running average of the first backbone's weights.

    See https://arxiv.org/abs/2006.07733 for more details (and please cite it if you
    use it in your own work).
    """

    def __init__(
        self,
        model: nn.Module,
        image_size: tuple[int, int] = (256, 256),
        hidden_layer: int = -2,
        in_channels: int = 4,
        projection_size: int = 256,
        hidden_size: int = 4096,
        augment_fn: nn.Module | None = None,
        beta: float = 0.99,
        **kwargs: Any,
    ) -> None:
        """Sets up a model for pre-training with BYOL using projection heads.

        Args:
            model: the model to pretrain using BYOL
            image_size: the size of the training images
            hidden_layer: the hidden layer in ``model`` to attach the projection
                head to, can be the name of the layer or index of the layer
            in_channels: number of input channels to the model
            projection_size: size of first layer of the projection MLP
            hidden_size: size of the hidden layer of the projection MLP
            augment_fn: an instance of a module that performs data augmentation
            beta: the speed at which the target backbone is updated using the main
                backbone
            **kwargs: Additional keyword arguments passed to :class:`nn.Module`
        """
        super().__init__()

        self.augment: nn.Module
        if augment_fn is None:
            self.augment = SimCLRAugmentation(image_size)
        else:
            self.augment = augment_fn

        self.beta = beta
        self.in_channels = in_channels
        self.backbone = BackboneWrapper(
            model, projection_size, hidden_size, layer=hidden_layer
        )
        self.predictor = MLP(projection_size, projection_size, hidden_size)
        self.target = BackboneWrapper(
            model, projection_size, hidden_size, layer=hidden_layer
        )

        # Perform a single forward pass to initialize the wrapper correctly
        self.backbone(torch.zeros(2, self.in_channels, *image_size))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the backbone model through the MLP and prediction head.

        Args:
            x: tensor of data to run through the model

        Returns:
            output from the model
        """
        z: Tensor = self.predictor(self.backbone(x))
        return z

    def update_target(self) -> None:
        """Method to update the "target" model weights."""
        for p, pt in zip(self.backbone.parameters(), self.target.parameters()):
            pt.data = self.beta * pt.data + (1 - self.beta) * p.data

class SimCLRAugmentation(nn.Module):
    """A module for applying SimCLR augmentations.

    SimCLR was one of the first papers to show the effectiveness of random data
    augmentation in self-supervised-learning setups. See
    https://arxiv.org/pdf/2002.05709.pdf for more details.
    """

    def __init__(self, image_size: tuple[int, int] = (256, 256)) -> None:
        """Initialize a module for applying SimCLR augmentations.

        Args:
            image_size: Tuple of integers defining the image size
        """
        super().__init__()
        self.size = image_size

        self.augmentation = nn.Sequential(
            K.Resize(size=image_size, align_corners=False),
            # Not suitable for multispectral adapt
            # K.ColorJitter(0.8, 0.8, 0.8, 0.8, 0.2),
            # K.RandomGrayscale(p=0.2),
            K.RandomHorizontalFlip(),
            K.RandomGaussianBlur((3, 3), (1.5, 1.5), p=0.1),
            K.RandomResizedCrop(size=image_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Applys SimCLR augmentations to the input tensor.

        Args:
            x: a batch of imagery

        Returns:
            an augmented batch of imagery
        """
        z: Tensor = self.augmentation(x)
        return z

