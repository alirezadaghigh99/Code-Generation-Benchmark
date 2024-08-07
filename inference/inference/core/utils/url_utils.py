def wrap_url(url: str) -> str:
    if not LICENSE_SERVER:
        return url
    return f"http://{LICENSE_SERVER}/proxy?url=" + urllib.parse.quote(
        url, safe="~()*!'"
    )

