def make_parallel_requests(
    requests_data: List[RequestData],
    request_method: RequestMethod,
) -> List[Response]:
    workers = len(requests_data)
    make_request_closure = partial(make_request, request_method=request_method)
    with ThreadPool(processes=workers) as pool:
        return pool.map(
            make_request_closure,
            iterable=requests_data,
        )