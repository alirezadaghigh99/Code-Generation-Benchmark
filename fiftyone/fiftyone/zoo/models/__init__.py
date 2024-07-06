def list_zoo_models(tags=None):
    """Returns the list of available models in the FiftyOne Model Zoo.

    Example usage::

        import fiftyone as fo
        import fiftyone.zoo as foz

        #
        # List all zoo models
        #

        names = foz.list_zoo_models()
        print(names)

        #
        # List all zoo models with the specified tag(s)
        #

        names = foz.list_zoo_models(tags="torch")
        print(names)

    Args:
        tags (None): only include models that have the specified tag or list
            of tags

    Returns:
        a list of model names
    """
    manifest = _load_zoo_models_manifest()

    if tags is not None:
        if etau.is_str(tags):
            tags = {tags}
        else:
            tags = set(tags)

        manifest = [model for model in manifest if tags.issubset(model.tags)]

    return sorted([model.name for model in manifest])

