def get_config_for_export_mode(should_be_onnx_standard: bool) -> NNCFConfig:
    nncf_config = NNCFConfig()
    nncf_config.update(
        {
            "input_info": {"sample_size": [1, 1, 4, 4]},
            "compression": {"algorithm": "quantization", "export_to_onnx_standard_ops": should_be_onnx_standard},
        }
    )
    register_bn_adaptation_init_args(nncf_config)
    return nncf_config

