def subclasses(cls: Type[Any]) -> Iterable[Type[Any]]:
    """
    Generator that returns all (not just direct) subclasses of `cls`
    """
    for subclass in cls.__subclasses__():
        yield from subclasses(subclass)
        yield subclass

