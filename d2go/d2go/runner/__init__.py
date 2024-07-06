def create_runner(
    class_full_name: Optional[str], *args, **kwargs
) -> Union[BaseRunner, Type[DefaultTask]]:
    """Constructs a runner instance if class is a d2go runner. Returns class
    type if class is a Lightning module.
    """
    if class_full_name is None:
        runner_class = GeneralizedRCNNRunner
    else:
        runner_class = import_runner(class_full_name)
    if issubclass(runner_class, DefaultTask):
        # Return runner class for Lightning module since it requires config
        # to construct
        return runner_class
    return runner_class(*args, **kwargs)

