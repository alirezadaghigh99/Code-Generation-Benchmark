def _determine_client_mode(api_url: str) -> HTTPClientMode:
    if any(api_url.startswith(roboflow_url) for roboflow_url in ALL_ROBOFLOW_API_URLS):
        return HTTPClientMode.V0
    return HTTPClientMode.V1