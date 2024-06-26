def api_key_safe_raise_for_status(response: Response) -> None:
    request_is_successful = response.status_code < 400
    if request_is_successful:
        return None
    response.url = API_KEY_PATTERN.sub(deduct_api_key, response.url)
    response.raise_for_status()