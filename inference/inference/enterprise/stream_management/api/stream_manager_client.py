def is_request_unsuccessful(response: dict) -> bool:
    return (
        response.get(RESPONSE_KEY, {}).get(STATUS_KEY, OperationStatus.FAILURE.value)
        != OperationStatus.SUCCESS.value
    )