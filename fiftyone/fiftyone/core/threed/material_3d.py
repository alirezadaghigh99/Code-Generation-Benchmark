class Material3D(BaseValidatedDataClass):
    """Base class for 3D materials.

    Args:
        opacity (1.0): the opacity of the material, in the range ``[0, 1]``
    """

    def __init__(self, opacity: float = 1.0):
        self.opacity = opacity

    @property
    def opacity(self) -> float:
        return self._opacity

    @opacity.setter
    def opacity(self, value: float) -> None:
        self._opacity = validate_float(value)

    def as_dict(self):
        return {
            "_type": self.__class__.__name__,
            "opacity": self.opacity,
        }

    @staticmethod
    def _from_dict(d):
        cls_name: str = d.pop("_type")
        if not cls_name.endswith("Material"):
            raise ValueError("Invalid material type")

        clz = getattr(threed, cls_name)
        return clz(**d)

class MeshMaterial(Material3D):
    """Represents a mesh material.

    Args:
        wireframe (False): whether to render the mesh as a wireframe
        opacity (1.0): the opacity of the material, in the range ``[0, 1]``
    """

    def __init__(self, wireframe: bool = False, opacity: float = 1.0):
        super().__init__(opacity)
        self.wireframe = wireframe

    @property
    def wireframe(self) -> bool:
        return self._wireframe

    @wireframe.setter
    def wireframe(self, value: bool) -> None:
        self._wireframe = validate_bool(value)

    def as_dict(self):
        return {**super().as_dict(), **{"wireframe": self.wireframe}}

