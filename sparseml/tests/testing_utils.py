def parse_params(
    configs_directory: str, type: Optional[str] = None
) -> List[Union[dict, CustomTestConfig]]:
    # parses the config file provided
    assert os.path.isdir(
        configs_directory
    ), f"Config_directory {configs_directory} is not a directory"

    config_dicts = []
    for file in os.listdir(configs_directory):
        config = _load_yaml(configs_directory, file)
        if not config:
            continue

        cadence = os.environ.get("CADENCE", "commit")
        expected_cadence = config.get("cadence")

        if not isinstance(expected_cadence, list):
            expected_cadence = [expected_cadence]
        if cadence in expected_cadence:
            if type == "custom":
                config = CustomTestConfig(**config)
            else:
                if not _validate_test_config(config):
                    raise ValueError(
                        "The config provided does not comply with the expected "
                        "structure. See tests.data.TestConfig for the expected "
                        "fields."
                    )
            config_dicts.append(config)
        else:
            logging.info(
                f"Skipping testing model: {file} for cadence: {config['cadence']}"
            )
    return config_dicts

