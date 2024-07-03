def save_text_lines_in_cache(
    content: List[str],
    file: str,
    model_id: Optional[str] = None,
    allow_override: bool = True,
) -> None:
    cached_file_path = get_cache_file_path(file=file, model_id=model_id)
    dump_text_lines(
        path=cached_file_path, content=content, allow_override=allow_override
    )