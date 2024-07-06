def through(cls, p0: Tensor, p1: Tensor) -> "ParametrizedLine":
        """Constructs a parametrized line going from a point :math:`p0` to :math:`p1`.

        Args:
            p0: tensor with first point :math:`(B, D)` where `D` is the point dimension.
            p1: tensor with second point :math:`(B, D)` where `D` is the point dimension.

        Example:
            >>> p0 = torch.tensor([0.0, 0.0])
            >>> p1 = torch.tensor([1.0, 1.0])
            >>> l = ParametrizedLine.through(p0, p1)
        """
        return ParametrizedLine(p0, normalize((p1 - p0), p=2, dim=-1))

