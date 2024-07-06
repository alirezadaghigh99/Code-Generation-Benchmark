def load_configs(config_path: Union[str, List[str]]) -> Dict:
    """
    Load multiple configs and update the nested dictionary.

    :param config_path: list of paths or one path.
    :return: the loaded config
    """
    if isinstance(config_path, str):
        config_path = [config_path]
    # replace ~ with user home path
    config_path = [os.path.expanduser(x) for x in config_path]
    config: Dict = {}
    for config_path_i in config_path:
        with open(config_path_i) as file:
            config_i = yaml.load(file, Loader=yaml.FullLoader)
        config = update_nested_dict(d=config, u=config_i)
    loaded_config = config_sanity_check(config)

    if loaded_config != config:
        # config got updated
        head, tail = os.path.split(config_path[0])
        filename = "updated_" + tail
        save(config=loaded_config, out_dir=head, filename=filename)
        logger.error(
            "The provided configuration file is outdated. "
            "An updated version has been saved at %s.",
            os.path.join(head, filename),
        )

    return loaded_config

