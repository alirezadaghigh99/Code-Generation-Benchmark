def load(
    model: str,
    repo: str = "onnx/models:main",
    opset: int | None = None,
    force_reload: bool = False,
    silent: bool = False,
) -> onnx.ModelProto | None:
    """Downloads a model by name from the onnx model hub.

    Args:
        model: The name of the onnx model in the manifest. This field is
            case-sensitive
        repo: The location of the model repo in format
            "user/repo[:branch]". If no branch is found will default to
            "main"
        opset: The opset of the model to download. The default of `None`
            automatically chooses the largest opset
        force_reload: Whether to force the model to re-download even if
            its already found in the cache
        silent: Whether to suppress the warning message if the repo is
            not trusted.

    Returns:
        ModelProto or None
    """
    selected_model = get_model_info(model, repo, opset)
    local_model_path_arr = selected_model.model_path.split("/")
    if selected_model.model_sha is not None:
        local_model_path_arr[-1] = (
            f"{selected_model.model_sha}_{local_model_path_arr[-1]}"
        )
    local_model_path = join(_ONNX_HUB_DIR, os.sep.join(local_model_path_arr))

    if force_reload or not os.path.exists(local_model_path):
        if not _verify_repo_ref(repo) and not silent:
            msg = f"The model repo specification {repo} is not trusted and may contain security vulnerabilities. Only continue if you trust this repo."

            print(msg, file=sys.stderr)
            print("Continue?[y/n]")
            if input().lower() != "y":
                return None

        os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
        lfs_url = _get_base_url(repo, True)
        print(f"Downloading {model} to local path {local_model_path}")
        _download_file(lfs_url + selected_model.model_path, local_model_path)
    else:
        print(f"Using cached {model} model from {local_model_path}")

    with open(local_model_path, "rb") as f:
        model_bytes = f.read()

    if selected_model.model_sha is not None:
        downloaded_sha = hashlib.sha256(model_bytes).hexdigest()
        if not downloaded_sha == selected_model.model_sha:
            raise AssertionError(
                f"The cached model {selected_model.model} has SHA256 {downloaded_sha} "
                f"while checksum should be {selected_model.model_sha}. "
                "The model in the hub may have been updated. Use force_reload to "
                "download the model from the model hub."
            )

    return onnx.load(cast(IO[bytes], BytesIO(model_bytes)))

