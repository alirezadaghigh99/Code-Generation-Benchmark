class RequestData:
    url: str
    request_elements: int
    headers: Optional[Dict[str, str]]
    parameters: Optional[Dict[str, Union[str, List[str]]]]
    data: Optional[Union[str, bytes]]
    payload: Optional[Dict[str, Any]]
    image_scaling_factors: List[Optional[float]]