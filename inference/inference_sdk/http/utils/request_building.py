def assembly_request_data(
    url: str,
    batch_inference_inputs: List[Tuple[str, Optional[float]]],
    headers: Optional[Dict[str, str]],
    parameters: Optional[Dict[str, Union[str, List[str]]]],
    payload: Optional[Dict[str, Any]],
    image_placement: ImagePlacement,
) -> RequestData:
    data = None
    if image_placement is ImagePlacement.DATA and len(batch_inference_inputs) != 1:
        raise ValueError("Only single image can be placed in request `data`")
    if image_placement is ImagePlacement.JSON and payload is None:
        payload = {}
    if image_placement is ImagePlacement.JSON:
        payload = deepcopy(payload)
        payload = inject_images_into_payload(
            payload=payload,
            encoded_images=batch_inference_inputs,
        )
    elif image_placement is ImagePlacement.DATA:
        data = batch_inference_inputs[0][0]
    else:
        raise NotImplemented(
            f"Not implemented request building method for {image_placement}"
        )
    scaling_factors = [e[1] for e in batch_inference_inputs]
    return RequestData(
        url=url,
        request_elements=len(batch_inference_inputs),
        headers=headers,
        parameters=parameters,
        data=data,
        payload=payload,
        image_scaling_factors=scaling_factors,
    )

