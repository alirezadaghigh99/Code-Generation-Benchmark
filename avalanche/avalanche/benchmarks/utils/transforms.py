class TupleTransform(MultiParamTransform):
    """Multi-argument transformation represented as tuples."""

    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = list(transforms)

    def __call__(self, *args):
        args_list = list(args)
        for idx, transform in enumerate(self.transforms):
            if transform is not None:
                args_list[idx] = transform(args_list[idx])
        return args_list

    def __str__(self):
        return "TupleTransform({})".format(self.transforms)

    def __repr__(self):
        return "TupleTransform({})".format(self.transforms)

    def __eq__(self, other):
        if self is other:
            return True

        if not isinstance(other, TupleTransform):
            return False

        return self.transforms == other.transforms

    def flat_transforms(self, position: int):
        if position < len(self.transforms):
            return flat_transforms_recursive(self.transforms[position], position)
        return []

