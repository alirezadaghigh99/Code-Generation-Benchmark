def create_compressed_model_and_algo_for_test(
    model: Module,
    config: NNCFConfig = None,
    dummy_forward_fn: Callable[[Module], Any] = None,
    wrap_inputs_fn: Callable[[Tuple, Dict], Tuple[Tuple, Dict]] = None,
    compression_state: Dict[str, Any] = None,
) -> Tuple[NNCFNetwork, PTCompressionAlgorithmController]:
    if config is not None:
        assert isinstance(config, NNCFConfig)
        NNCFConfig.validate(config)
    algo, model = create_compressed_model(
        model,
        config,
        dump_graphs=False,
        dummy_forward_fn=dummy_forward_fn,
        wrap_inputs_fn=wrap_inputs_fn,
        compression_state=compression_state,
    )
    return model, algo

def set_torch_seed(seed: int = 42):
    saved_seed = torch.seed()
    torch.manual_seed(seed)
    yield
    torch.manual_seed(saved_seed)

def get_empty_config(
    model_size=4, input_sample_sizes: Union[Tuple[List[int]], List[int]] = None, input_info: Dict = None
) -> NNCFConfig:
    if input_sample_sizes is None:
        input_sample_sizes = [1, 1, 4, 4]

    def _create_input_info():
        if isinstance(input_sample_sizes, tuple):
            return [{"sample_size": sizes} for sizes in input_sample_sizes]
        return [{"sample_size": input_sample_sizes}]

    config = NNCFConfig()
    config.update(
        {
            "model": "empty_config",
            "model_size": model_size,
            "input_info": input_info if input_info else _create_input_info(),
        }
    )
    return config

