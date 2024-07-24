class AmbientLights(TensorProperties):
    """
    A light object representing the same color of light everywhere.
    By default, this is white, which effectively means lighting is
    not used in rendering.

    Unlike other lights this supports an arbitrary number of channels, not just 3 for RGB.
    The ambient_color input determines the number of channels.
    """

    def __init__(self, *, ambient_color=None, device: Device = "cpu") -> None:
        """
        If ambient_color is provided, it should be a sequence of
        triples of floats.

        Args:
            ambient_color: RGB color
            device: Device (as str or torch.device) on which the tensors should be located

        The ambient_color if provided, should be
            - tuple/list of C-element tuples of floats
            - torch tensor of shape (1, C)
            - torch tensor of shape (N, C)
        where C is the number of channels and N is batch size.
        For RGB, C is 3.
        """
        if ambient_color is None:
            ambient_color = ((1.0, 1.0, 1.0),)
        super().__init__(ambient_color=ambient_color, device=device)

    def clone(self):
        other = self.__class__(device=self.device)
        return super().clone(other)

    def diffuse(self, normals, points) -> torch.Tensor:
        return self._zeros_channels(points)

    def specular(self, normals, points, camera_position, shininess) -> torch.Tensor:
        return self._zeros_channels(points)

    def _zeros_channels(self, points: torch.Tensor) -> torch.Tensor:
        ch = self.ambient_color.shape[-1]
        return torch.zeros(*points.shape[:-1], ch, device=points.device)

class DirectionalLights(TensorProperties):
    def __init__(
        self,
        ambient_color=((0.5, 0.5, 0.5),),
        diffuse_color=((0.3, 0.3, 0.3),),
        specular_color=((0.2, 0.2, 0.2),),
        direction=((0, 1, 0),),
        device: Device = "cpu",
    ) -> None:
        """
        Args:
            ambient_color: RGB color of the ambient component.
            diffuse_color: RGB color of the diffuse component.
            specular_color: RGB color of the specular component.
            direction: (x, y, z) direction vector of the light.
            device: Device (as str or torch.device) on which the tensors should be located

        The inputs can each be
            - 3 element tuple/list or list of lists
            - torch tensor of shape (1, 3)
            - torch tensor of shape (N, 3)
        The inputs are broadcast against each other so they all have batch
        dimension N.
        """
        super().__init__(
            device=device,
            ambient_color=ambient_color,
            diffuse_color=diffuse_color,
            specular_color=specular_color,
            direction=direction,
        )
        _validate_light_properties(self)
        if self.direction.shape[-1] != 3:
            msg = "Expected direction to have shape (N, 3); got %r"
            raise ValueError(msg % repr(self.direction.shape))

    def clone(self):
        other = self.__class__(device=self.device)
        return super().clone(other)

    def diffuse(self, normals, points=None) -> torch.Tensor:
        # NOTE: Points is not used but is kept in the args so that the API is
        # the same for directional and point lights. The call sites should not
        # need to know the light type.
        return diffuse(
            normals=normals,
            color=self.diffuse_color,
            direction=self.direction,
        )

    def specular(self, normals, points, camera_position, shininess) -> torch.Tensor:
        return specular(
            points=points,
            normals=normals,
            color=self.specular_color,
            direction=self.direction,
            camera_position=camera_position,
            shininess=shininess,
        )

class PointLights(TensorProperties):
    def __init__(
        self,
        ambient_color=((0.5, 0.5, 0.5),),
        diffuse_color=((0.3, 0.3, 0.3),),
        specular_color=((0.2, 0.2, 0.2),),
        location=((0, 1, 0),),
        device: Device = "cpu",
    ) -> None:
        """
        Args:
            ambient_color: RGB color of the ambient component
            diffuse_color: RGB color of the diffuse component
            specular_color: RGB color of the specular component
            location: xyz position of the light.
            device: Device (as str or torch.device) on which the tensors should be located

        The inputs can each be
            - 3 element tuple/list or list of lists
            - torch tensor of shape (1, 3)
            - torch tensor of shape (N, 3)
        The inputs are broadcast against each other so they all have batch
        dimension N.
        """
        super().__init__(
            device=device,
            ambient_color=ambient_color,
            diffuse_color=diffuse_color,
            specular_color=specular_color,
            location=location,
        )
        _validate_light_properties(self)
        if self.location.shape[-1] != 3:
            msg = "Expected location to have shape (N, 3); got %r"
            raise ValueError(msg % repr(self.location.shape))

    def clone(self):
        other = self.__class__(device=self.device)
        return super().clone(other)

    def reshape_location(self, points) -> torch.Tensor:
        """
        Reshape the location tensor to have dimensions
        compatible with the points which can either be of
        shape (P, 3) or (N, H, W, K, 3).
        """
        if self.location.ndim == points.ndim:
            return self.location
        return self.location[:, None, None, None, :]

    def diffuse(self, normals, points) -> torch.Tensor:
        location = self.reshape_location(points)
        direction = location - points
        return diffuse(normals=normals, color=self.diffuse_color, direction=direction)

    def specular(self, normals, points, camera_position, shininess) -> torch.Tensor:
        location = self.reshape_location(points)
        direction = location - points
        return specular(
            points=points,
            normals=normals,
            color=self.specular_color,
            direction=direction,
            camera_position=camera_position,
            shininess=shininess,
        )

