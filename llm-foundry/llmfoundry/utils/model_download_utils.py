def download_from_hf_hub(
    model: str,
    save_dir: str,
    prefer_safetensors: bool = True,
    tokenizer_only: bool = False,
    token: Optional[str] = None,
):
    """Downloads model files from a Hugging Face Hub model repo.

    Only supports models stored in Safetensors and PyTorch formats for now. If both formats are available, only the
    Safetensors weights will be downloaded unless `prefer_safetensors` is set to False.

    Args:
        repo_id (str): The Hugging Face Hub repo ID.
        save_dir (str, optional): The local path to the directory where the model files will be downloaded.
        prefer_safetensors (bool): Whether to prefer Safetensors weights over PyTorch weights if both are
            available. Defaults to True.
        tokenizer_only (bool): If true, only download tokenizer files.
        token (str, optional): The HuggingFace API token. If not provided, the token will be read from the
            `HF_TOKEN` environment variable.

    Raises:
        RepositoryNotFoundError: If the model repo doesn't exist or the token is unauthorized.
        ValueError: If the model repo doesn't contain any supported model weights.
    """
    repo_files = set(hf_hub.list_repo_files(model))

    # Ignore TensorFlow, TensorFlow 2, and Flax weights as they are not supported by Composer.
    ignore_patterns = copy.deepcopy(DEFAULT_IGNORE_PATTERNS)

    safetensors_available = (
        SAFE_WEIGHTS_NAME in repo_files or SAFE_WEIGHTS_INDEX_NAME in repo_files
    )
    pytorch_available = (
        PYTORCH_WEIGHTS_NAME in repo_files or
        PYTORCH_WEIGHTS_INDEX_NAME in repo_files
    )

    if safetensors_available and pytorch_available:
        if prefer_safetensors:
            log.info(
                'Safetensors available and preferred. Excluding pytorch weights.',
            )
            ignore_patterns.append(PYTORCH_WEIGHTS_PATTERN)
        else:
            log.info(
                'Pytorch available and preferred. Excluding safetensors weights.',
            )
            ignore_patterns.append(SAFE_WEIGHTS_PATTERN)
    elif safetensors_available:
        log.info('Only safetensors available. Ignoring weights preference.')
    elif pytorch_available:
        log.info('Only pytorch available. Ignoring weights preference.')
    else:
        raise ValueError(
            f'No supported model weights found in repo {model}.' +
            ' Please make sure the repo contains either safetensors or pytorch weights.',
        )

    allow_patterns = TOKENIZER_FILES if tokenizer_only else None

    download_start = time.time()
    hf_hub.snapshot_download(
        model,
        local_dir=save_dir,
        local_dir_use_symlinks=False,
        ignore_patterns=ignore_patterns,
        allow_patterns=allow_patterns,
        token=token,
    )
    download_duration = time.time() - download_start
    log.info(
        f'Downloaded model {model} from Hugging Face Hub in {download_duration} seconds',
    )