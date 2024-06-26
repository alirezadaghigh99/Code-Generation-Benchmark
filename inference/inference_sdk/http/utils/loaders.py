def uri_is_http_link(uri: str) -> bool:
    return uri.startswith("http://") or uri.startswith("https://")