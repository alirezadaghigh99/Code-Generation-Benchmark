def send_data_trough_socket(
    target: socket.socket,
    header_size: int,
    data: bytes,
    request_id: str,
    recover_from_overflow: bool = True,
    pipeline_id: Optional[str] = None,
) -> None:
    try:
        data_size = len(data)
        header = data_size.to_bytes(length=header_size, byteorder="big")
        payload = header + data
        target.sendall(payload)
    except OverflowError as error:
        if not recover_from_overflow:
            logger.error(f"OverflowError was suppressed. {error}")
            return None
        error_response = prepare_error_response(
            request_id=request_id,
            error=error,
            error_type=ErrorType.INTERNAL_ERROR,
            pipeline_id=pipeline_id,
        )
        send_data_trough_socket(
            target=target,
            header_size=header_size,
            data=error_response,
            request_id=request_id,
            recover_from_overflow=False,
            pipeline_id=pipeline_id,
        )
    except Exception as error:
        logger.error(f"Could not send the response through socket. Error: {error}")