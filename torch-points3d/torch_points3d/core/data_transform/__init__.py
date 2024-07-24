def instantiate_transforms(transform_options):
    """ Creates a torch_geometric composite transform from an OmegaConf list such as
    - transform: GridSampling3D
        params:
            size: 0.01
    - transform: NormaliseScale
    """
    transforms = []
    for transform in transform_options:
        transforms.append(instantiate_transform(transform))
    return T.Compose(transforms)

class LotteryTransform(object):
    """
    Transforms which draw a transform randomly among several transforms indicated in transform options
    Examples

    Parameters
    ----------
    transform_options Omegaconf list which contains the transform
    """

    def __init__(self, transform_options):
        self.random_transforms = instantiate_transforms(transform_options)

    def __call__(self, data):

        list_transforms = self.random_transforms.transforms
        i = np.random.randint(len(list_transforms))
        transform = list_transforms[i]
        return transform(data)

    def __repr__(self):
        rep = "LotteryTransform(["
        for trans in self.random_transforms.transforms:
            rep = rep + "{}, ".format(trans.__repr__())
        rep = rep + "])"
        return rep

