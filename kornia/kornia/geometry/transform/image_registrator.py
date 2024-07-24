class Homography(BaseModel):
    r"""Homography geometric model to be used together with ImageRegistrator module for the optimization-based
    image registration."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Parameter(torch.eye(3))
        self.reset_model()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.model})"

    def reset_model(self) -> None:
        """Initializes the model with identity transform."""
        torch.nn.init.eye_(self.model)

    def forward(self) -> Tensor:
        r"""Single-batch homography".

        Returns:
            Homography matrix with shape :math:`(1, 3, 3)`.
        """
        return torch.unsqueeze(self.model / self.model[2, 2], dim=0)  # 1x3x3

    def forward_inverse(self) -> Tensor:
        r"""Interted Single-batch homography".

        Returns:
            Homography martix with shape :math:`(1, 3, 3)`.
        """
        return torch.unsqueeze(torch.inverse(self.model), dim=0)

