def dispatch_error(error_response: dict) -> None:
    response_payload = error_response.get(RESPONSE_KEY, {})
    error_type = response_payload.get(ERROR_TYPE_KEY)
    error_class = response_payload.get("error_class", "N/A")
    error_message = response_payload.get("error_message", "N/A")
    logger.error(
        f"Error in ProcessesManagerClient. error_type={error_type} error_class={error_class} "
        f"error_message={error_message}"
    )
    if error_type in ERRORS_MAPPING:
        raise ERRORS_MAPPING[error_type](
            f"Error in ProcessesManagerClient. Error type: {error_type}. Details: {error_message}"
        )
    raise ProcessesManagerClientError(
        f"Error in ProcessesManagerClient. Error type: {error_type}. Details: {error_message}"
    )

def is_request_unsuccessful(response: dict) -> bool:
    return (
        response.get(RESPONSE_KEY, {}).get(STATUS_KEY, OperationStatus.FAILURE.value)
        != OperationStatus.SUCCESS.value
    )

