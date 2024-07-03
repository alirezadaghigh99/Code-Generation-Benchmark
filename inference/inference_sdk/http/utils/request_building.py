def prepare_requests_data(
    url: str,
    encoded_inference_inputs: List[Tuple[str, Optional[float]]],
    headers: Optional[Dict[str, str]],
    parameters: Optional[Dict[str, Union[str, List[str]]]],
    payload: Optional[Dict[str, Any]],
    max_batch_size: int,
    image_placement: ImagePlacement,
) -> List[RequestData]:
    batches = list(
        make_batches(
            iterable=encoded_inference_inputs,
            batch_size=max_batch_size,
        )
    )
    requests_data = []
    for batch_inference_inputs in batches:
        request_data = assembly_request_data(
            url=url,
            batch_inference_inputs=batch_inference_inputs,
            headers=headers,
            parameters=parameters,
            payload=payload,
            image_placement=image_placement,
        )
        requests_data.append(request_data)
    return requests_data