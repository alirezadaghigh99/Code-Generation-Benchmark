def are_all_files_cached(
    files: List[Union[str, re.Pattern]], model_id: Optional[str] = None
) -> bool:
    return all(is_file_cached(file=file, model_id=model_id) for file in files)

def is_file_cached(
    file: Union[str, re.Pattern], model_id: Optional[str] = None
) -> bool:
    if isinstance(file, re.Pattern):
        return exists_file_matching_regex(file, model_id=model_id)

    cached_file_path = get_cache_file_path(file=file, model_id=model_id)
    return os.path.isfile(cached_file_path)

def get_cache_file_path(file: str, model_id: Optional[str] = None) -> str:
    cache_dir = get_cache_dir(model_id=model_id)
    return os.path.join(cache_dir, file)

def get_cache_dir(model_id: Optional[str] = None) -> str:
    if model_id is not None:
        return os.path.join(MODEL_CACHE_DIR, model_id)
    return MODEL_CACHE_DIR

def clear_cache(model_id: Optional[str] = None) -> None:
    cache_dir = get_cache_dir(model_id=model_id)
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

