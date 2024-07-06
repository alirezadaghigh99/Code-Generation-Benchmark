def get_basic_pruning_config(input_sample_size=None) -> NNCFConfig:
    if input_sample_size is None:
        input_sample_size = [1, 1, 4, 4]
    config = NNCFConfig()
    config.update(
        {
            "model": "pruning_conv_model",
            "input_info": {
                "sample_size": input_sample_size,
            },
            "compression": {"params": {}},
        }
    )
    return config

