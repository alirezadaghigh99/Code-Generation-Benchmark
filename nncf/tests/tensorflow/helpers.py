def create_compressed_model_and_algo_for_test(model, config, compression_state=None, force_no_init=False):
    assert isinstance(config, NNCFConfig)
    tf.keras.backend.clear_session()
    if force_no_init:
        compression_state = {BaseCompressionAlgorithmController.BUILDER_STATE: {}}
    algo, model = create_compressed_model(model, config, compression_state)
    return model, algo

def get_empty_config(input_sample_sizes=None) -> NNCFConfig:
    if input_sample_sizes is None:
        input_sample_sizes = [1, 4, 4, 1]

    def _create_input_info():
        if isinstance(input_sample_sizes, tuple):
            return [{"sample_size": sizes} for sizes in input_sample_sizes]
        return [{"sample_size": input_sample_sizes}]

    config = NNCFConfig({"model": "basic_sparse_conv", "input_info": _create_input_info()})
    return config

