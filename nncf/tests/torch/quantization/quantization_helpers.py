def get_quantization_config_without_range_init(model_size=4) -> NNCFConfig:
    config = get_empty_config(input_sample_sizes=[1, 1, model_size, model_size])
    config["compression"] = {"algorithm": "quantization", "initializer": {"range": {"num_init_samples": 0}}}
    return config

