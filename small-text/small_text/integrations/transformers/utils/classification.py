def _get_arguments_for_from_pretrained_model(model_loading_strategy: ModelLoadingStrategy) \
        -> PretrainedModelLoadingArguments:

    if model_loading_strategy == ModelLoadingStrategy.DEFAULT:
        if str(os.environ.get('TRANSFORMERS_OFFLINE', '0')) == '1':
            # same as ALWAYS_LOCAL
            return PretrainedModelLoadingArguments(local_files_only=True)
        else:
            return PretrainedModelLoadingArguments()
    elif model_loading_strategy == ModelLoadingStrategy.ALWAYS_LOCAL:
        return PretrainedModelLoadingArguments(local_files_only=True)
    else:
        return PretrainedModelLoadingArguments(force_download=True)

