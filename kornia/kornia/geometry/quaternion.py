def random(
        cls, batch_size: Optional[int] = None, device: Optional[Device] = None, dtype: Dtype = None
    ) -> "Quaternion":
        """Create a random unit quaternion of shape :math:`(B, 4)`.

        Uniformly distributed across the rotation space as per: http://planning.cs.uiuc.edu/node198.html

        Args:
            batch_size: the batch size of the underlying data.

        Example:
            >>> q = Quaternion.random()
            >>> q = Quaternion.random(batch_size=2)
        """
        rand_shape = (batch_size,) if batch_size is not None else ()

        r1, r2, r3 = rand((3, *rand_shape), device=device, dtype=dtype)
        q1 = (1.0 - r1).sqrt() * ((2 * pi * r2).sin())
        q2 = (1.0 - r1).sqrt() * ((2 * pi * r2).cos())
        q3 = r1.sqrt() * (2 * pi * r3).sin()
        q4 = r1.sqrt() * (2 * pi * r3).cos()
        return cls(stack((q1, q2, q3, q4), -1))

def from_coeffs(cls, w: float, x: float, y: float, z: float) -> "Quaternion":
        """Create a quaternion from the data coefficients.

        Args:
            w: a float representing the :math:`q_w` component.
            x: a float representing the :math:`q_x` component.
            y: a float representing the :math:`q_y` component.
            z: a float representing the :math:`q_z` component.

        Example:
            >>> q = Quaternion.from_coeffs(1., 0., 0., 0.)
            >>> q.data
            Parameter containing:
            tensor([1., 0., 0., 0.], requires_grad=True)
        """
        return cls(tensor([w, x, y, z]))

