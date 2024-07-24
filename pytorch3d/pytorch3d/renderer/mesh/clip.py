class ClipFrustum:
    """
    Helper class to store the information needed to represent a view frustum
    (left, right, top, bottom, znear, zfar), which is used to clip or cull triangles.
    Values left as None mean that culling should not be performed for that axis.
    The parameters perspective_correct, cull, and z_clip_value are used to define
    behavior for clipping triangles to the frustum.

    Args:
        left: NDC coordinate of the left clipping plane (along x axis)
        right: NDC coordinate of the right clipping plane (along x axis)
        top: NDC coordinate of the top clipping plane (along y axis)
        bottom: NDC coordinate of the bottom clipping plane (along y axis)
        znear: world space z coordinate of the near clipping plane
        zfar: world space z coordinate of the far clipping plane
        perspective_correct: should be set to True for a perspective camera
        cull: if True, triangles outside the frustum should be culled
        z_clip_value: if not None, then triangles should be clipped (possibly into
            smaller triangles) such that z >= z_clip_value.  This avoids projections
            that go to infinity as z->0
    """

    __slots__ = [
        "left",
        "right",
        "top",
        "bottom",
        "znear",
        "zfar",
        "perspective_correct",
        "cull",
        "z_clip_value",
    ]

    def __init__(
        self,
        left: Optional[float] = None,
        right: Optional[float] = None,
        top: Optional[float] = None,
        bottom: Optional[float] = None,
        znear: Optional[float] = None,
        zfar: Optional[float] = None,
        perspective_correct: bool = False,
        cull: bool = True,
        z_clip_value: Optional[float] = None,
    ) -> None:
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.znear = znear
        self.zfar = zfar
        self.perspective_correct = perspective_correct
        self.cull = cull
        self.z_clip_value = z_clip_value

