def clear_cache(model_id: Optional[str] = None) -> None:
    cache_dir = get_cache_dir(model_id=model_id)
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)