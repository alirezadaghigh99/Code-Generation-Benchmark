def get_response_ignoring_thrash(
    responses_queue: Queue, matching_request_id: str
) -> dict:
    while True:
        response = responses_queue.get()
        if response[0] == matching_request_id:
            return response[1]
        logger.warning(
            f"Dropping response for request_id={response[0]} with payload={response[1]}"
        )

def handle_command(
    processes_table: Dict[str, Tuple[Process, Queue, Queue]],
    request_id: str,
    pipeline_id: str,
    command: dict,
) -> dict:
    if pipeline_id not in processes_table:
        return describe_error(exception=None, error_type=ErrorType.NOT_FOUND)
    _, command_queue, responses_queue = processes_table[pipeline_id]
    command_queue.put((request_id, command))
    return get_response_ignoring_thrash(
        responses_queue=responses_queue, matching_request_id=request_id
    )

