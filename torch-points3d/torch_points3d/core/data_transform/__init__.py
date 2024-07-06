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

