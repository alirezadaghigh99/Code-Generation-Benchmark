def load_json_from_cache(
    file: str, model_id: Optional[str] = None, **kwargs
) -> Optional[Union[dict, list]]:
    cached_file_path = get_cache_file_path(file=file, model_id=model_id)
    return read_json(path=cached_file_path, **kwargs)