class MinimalDataParserConfig(DataParserConfig):
    """Minimal dataset config"""

    _target: Type = field(default_factory=lambda: MinimalDataParser)
    """target class to instantiate"""
    data: Path = Path("/home/nikhil/nerfstudio-main/tests/data/lego_test/minimal_parser")

