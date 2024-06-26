def _find_free_source_identifier(videos: List[Union[VideoSource, str, int]]) -> int:
    minimal_free_source_id = [
        v.source_id if v.source_id is not None else -1
        for v in videos
        if issubclass(type(v), VideoSource)
    ]
    if len(minimal_free_source_id) == 0:
        minimal_free_source_id = -1
    else:
        minimal_free_source_id = max(minimal_free_source_id)
    minimal_free_source_id += 1
    return minimal_free_source_id