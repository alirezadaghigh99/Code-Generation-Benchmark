def push_to_hf_hub(model: Any, model_name: str, task: str, **kwargs) -> None:  # pragma: no cover
    """Save model and its configuration on HF hub

    >>> from doctr.models import login_to_hub, push_to_hf_hub
    >>> from doctr.models.recognition import crnn_mobilenet_v3_small
    >>> login_to_hub()
    >>> model = crnn_mobilenet_v3_small(pretrained=True)
    >>> push_to_hf_hub(model, 'my-model', 'recognition', arch='crnn_mobilenet_v3_small')

    Args:
    ----
        model: TF or PyTorch model to be saved
        model_name: name of the model which is also the repository name
        task: task name
        **kwargs: keyword arguments for push_to_hf_hub
    """
    run_config = kwargs.get("run_config", None)
    arch = kwargs.get("arch", None)

    if run_config is None and arch is None:
        raise ValueError("run_config or arch must be specified")
    if task not in ["classification", "detection", "recognition"]:
        raise ValueError("task must be one of classification, detection, recognition")

    # default readme
    readme = textwrap.dedent(
        f"""
    ---
    language: en
    ---

    <p align="center">
    <img src="https://doctr-static.mindee.com/models?id=v0.3.1/Logo_doctr.gif&src=0" width="60%">
    </p>

    **Optical Character Recognition made seamless & accessible to anyone, powered by TensorFlow 2 & PyTorch**

    ## Task: {task}

    https://github.com/mindee/doctr

    ### Example usage:

    ```python
    >>> from doctr.io import DocumentFile
    >>> from doctr.models import ocr_predictor, from_hub

    >>> img = DocumentFile.from_images(['<image_path>'])
    >>> # Load your model from the hub
    >>> model = from_hub('mindee/my-model')

    >>> # Pass it to the predictor
    >>> # If your model is a recognition model:
    >>> predictor = ocr_predictor(det_arch='db_mobilenet_v3_large',
    >>>                           reco_arch=model,
    >>>                           pretrained=True)

    >>> # If your model is a detection model:
    >>> predictor = ocr_predictor(det_arch=model,
    >>>                           reco_arch='crnn_mobilenet_v3_small',
    >>>                           pretrained=True)

    >>> # Get your predictions
    >>> res = predictor(img)
    ```
    """
    )

    # add run configuration to readme if available
    if run_config is not None:
        arch = run_config.arch
        readme += textwrap.dedent(
            f"""### Run Configuration
                                  \n{json.dumps(vars(run_config), indent=2, ensure_ascii=False)}"""
        )

    if arch not in AVAILABLE_ARCHS[task]:
        raise ValueError(
            f"Architecture: {arch} for task: {task} not found.\
                         \nAvailable architectures: {AVAILABLE_ARCHS}"
        )

    commit_message = f"Add {model_name} model"

    local_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", model_name)
    repo_url = HfApi().create_repo(model_name, token=get_token(), exist_ok=False)
    repo = Repository(local_dir=local_cache_dir, clone_from=repo_url, use_auth_token=True)

    with repo.commit(commit_message):
        _save_model_and_config_for_hf_hub(model, repo.local_dir, arch=arch, task=task)
        readme_path = Path(repo.local_dir) / "README.md"
        readme_path.write_text(readme)

    repo.git_push()