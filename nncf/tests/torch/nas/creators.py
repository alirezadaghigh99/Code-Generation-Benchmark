def create_single_conv_kernel_supernet(
    kernel_size=5, out_channels=1, padding=2
) -> Tuple[ElasticKernelHandler, NNCFNetwork]:
    params = {"available_elasticity_dims": [ElasticityDim.KERNEL.value]}
    model_creator = partial(BasicConvTestModel, 1, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
    input_sample_sizes = [1, 1, kernel_size, kernel_size]
    multi_elasticity_handler, supernet = create_supernet(model_creator, input_sample_sizes, params)
    move_model_to_cuda_if_available(supernet)
    return multi_elasticity_handler.kernel_handler, supernet

def create_two_conv_width_supernet(elasticity_params=None, model=TwoConvModel):
    params = {"available_elasticity_dims": [ElasticityDim.WIDTH.value]}
    if elasticity_params is not None:
        params.update(elasticity_params)
    multi_elasticity_handler, supernet = create_supernet(model, model.INPUT_SIZE, params)
    move_model_to_cuda_if_available(supernet)
    return multi_elasticity_handler.width_handler, supernet

def create_bnas_model_and_ctrl_by_test_desc(desc: MultiElasticityTestDesc):
    config = {
        "input_info": {"sample_size": desc.input_sizes},
        "bootstrapNAS": {"training": {"elasticity": {"depth": {"skipped_blocks": desc.blocks_to_skip}}}},
    }
    depth_config = config["bootstrapNAS"]["training"]["elasticity"]["depth"]
    if not desc.blocks_to_skip:
        del depth_config["skipped_blocks"]
    config["bootstrapNAS"]["training"]["elasticity"].update(desc.algo_params)

    nncf_config = NNCFConfig.from_dict(config)
    model = desc.model_creator()
    model.eval()
    move_model_to_cuda_if_available(model)
    model, training_ctrl = create_bootstrap_training_model_and_ctrl(model, nncf_config)
    return model, training_ctrl

def create_bootstrap_nas_training_algo(model_name) -> Tuple[NNCFNetwork, ProgressiveShrinkingController, Callable]:
    model = NAS_MODEL_DESCS[model_name][0]()
    nncf_config = get_empty_config(input_sample_sizes=NAS_MODEL_DESCS[model_name][1])
    nncf_config["bootstrapNAS"] = {"training": {"algorithm": "progressive_shrinking"}}
    nncf_config["input_info"][0].update({"filler": "random"})

    input_info = FillerInputInfo.from_nncf_config(nncf_config)
    dummy_forward = create_dummy_forward_fn(input_info)
    compressed_model, training_ctrl = create_bootstrap_training_model_and_ctrl(model, nncf_config)
    return compressed_model, training_ctrl, dummy_forward

