class BlenderDataParserConfig(DataParserConfig):
    """Blender dataset parser config"""

    _target: Type = field(default_factory=lambda: Blender)
    """target class to instantiate"""
    data: Path = Path("data/blender/lego")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    alpha_color: Optional[str] = "white"
    """alpha color of background, when set to None, InputDataset that consumes DataparserOutputs will not attempt 
    to blend with alpha_colors using image's alpha channel data. Thus rgba image will be directly used in training. """
    ply_path: Optional[Path] = None
    """Path to PLY file to load 3D points from, defined relative to the dataset directory. This is helpful for
    Gaussian splatting and generally unused otherwise. If `None`, points are initialized randomly."""

