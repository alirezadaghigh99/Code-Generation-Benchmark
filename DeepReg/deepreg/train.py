def build_config(
    config_path: Union[str, List[str]],
    log_dir: str,
    exp_name: str,
    ckpt_path: str,
    max_epochs: int = -1,
) -> Tuple[Dict, str, str]:
    """
    Function to initialise log directories,
    assert that checkpointed model is the right
    type and to parse the configuration for training.

    :param config_path: list of str, path to config file
    :param log_dir: path of the log directory
    :param exp_name: name of the experiment
    :param ckpt_path: path where model is stored.
    :param max_epochs: if max_epochs > 0, use it to overwrite the configuration
    :return: - config: a dictionary saving configuration
             - exp_name: the path of directory to save logs
    """

    # init log directory
    log_dir = build_log_dir(log_dir=log_dir, exp_name=exp_name)

    # load config
    config = config_parser.load_configs(config_path)

    # replace the ~ with user home path
    ckpt_path = os.path.expanduser(ckpt_path)

    # overwrite epochs and save_period if necessary
    if max_epochs > 0:
        config["train"]["epochs"] = max_epochs
        config["train"]["save_period"] = min(max_epochs, config["train"]["save_period"])

    # backup config
    config_parser.save(config=config, out_dir=log_dir)

    return config, log_dir, ckpt_path

