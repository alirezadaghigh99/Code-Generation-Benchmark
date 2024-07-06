def assert_request_equal(request, dictionary):
    for method, requests in dictionary.items():
        mmr = getattr(request, method)
        assert mmr.requests == requests

    empty_methods = [method for method in SIMPLE_METHODS if method not in dictionary]
    for method in empty_methods:
        assert not len(getattr(request, method).requests)

def assert_request_is_empty(metadata_request, exclude=None):
    """Check if a metadata request dict is empty.

    One can exclude a method or a list of methods from the check using the
    ``exclude`` parameter. If metadata_request is a MetadataRouter, then
    ``exclude`` can be of the form ``{"object" : [method, ...]}``.
    """
    if isinstance(metadata_request, MetadataRouter):
        for name, route_mapping in metadata_request:
            if exclude is not None and name in exclude:
                _exclude = exclude[name]
            else:
                _exclude = None
            assert_request_is_empty(route_mapping.router, exclude=_exclude)
        return

    exclude = [] if exclude is None else exclude
    for method in SIMPLE_METHODS:
        if method in exclude:
            continue
        mmr = getattr(metadata_request, method)
        props = [
            prop
            for prop, alias in mmr.requests.items()
            if isinstance(alias, str) or alias is not None
        ]
        assert not props

