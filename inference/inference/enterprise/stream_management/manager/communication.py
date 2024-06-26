def receive_socket_data(
    source: socket.socket, header_size: int, buffer_size: int
) -> dict:
    header = source.recv(header_size)
    if len(header) != header_size:
        raise MalformedHeaderError(
            f"Expected header size: {header_size}, received: {header}"
        )
    payload_size = int.from_bytes(bytes=header, byteorder="big")
    if payload_size <= 0:
        raise MalformedHeaderError(
            f"Header is indicating non positive payload size: {payload_size}"
        )
    received = b""
    while len(received) < payload_size:
        chunk = source.recv(buffer_size)
        if len(chunk) == 0:
            raise TransmissionChannelClosed(
                "Socket was closed to read before payload was decoded."
            )
        received += chunk
    try:
        return json.loads(received)
    except ValueError:
        raise MalformedPayloadError("Received payload that is not in a JSON format")