def process_routing(_obj, _method, /, **kwargs):
    """Validate and route input parameters.

    This function is used inside a router's method, e.g. :term:`fit`,
    to validate the metadata and handle the routing.

    Assuming this signature of a router's fit method:
    ``fit(self, X, y, sample_weight=None, **fit_params)``,
    a call to this function would be:
    ``process_routing(self, "fit", sample_weight=sample_weight, **fit_params)``.

    Note that if routing is not enabled and ``kwargs`` is empty, then it
    returns an empty routing where ``process_routing(...).ANYTHING.ANY_METHOD``
    is always an empty dictionary.

    .. versionadded:: 1.3

    Parameters
    ----------
    _obj : object
        An object implementing ``get_metadata_routing``. Typically a
        meta-estimator.

    _method : str
        The name of the router's method in which this function is called.

    **kwargs : dict
        Metadata to be routed.

    Returns
    -------
    routed_params : Bunch
        A :class:`~utils.Bunch` of the form ``{"object_name": {"method_name":
        {params: value}}}`` which can be used to pass the required metadata to
        A :class:`~sklearn.utils.Bunch` of the form ``{"object_name": {"method_name":
        {params: value}}}`` which can be used to pass the required metadata to
        corresponding methods or corresponding child objects. The object names
        are those defined in `obj.get_metadata_routing()`.
    """
    if not kwargs:
        # If routing is not enabled and kwargs are empty, then we don't have to
        # try doing any routing, we can simply return a structure which returns
        # an empty dict on routed_params.ANYTHING.ANY_METHOD.
        class EmptyRequest:
            def get(self, name, default=None):
                return Bunch(**{method: dict() for method in METHODS})

            def __getitem__(self, name):
                return Bunch(**{method: dict() for method in METHODS})

            def __getattr__(self, name):
                return Bunch(**{method: dict() for method in METHODS})

        return EmptyRequest()

    if not (hasattr(_obj, "get_metadata_routing") or isinstance(_obj, MetadataRouter)):
        raise AttributeError(
            f"The given object ({repr(_obj.__class__.__name__)}) needs to either"
            " implement the routing method `get_metadata_routing` or be a"
            " `MetadataRouter` instance."
        )
    if _method not in METHODS:
        raise TypeError(
            f"Can only route and process input on these methods: {METHODS}, "
            f"while the passed method is: {_method}."
        )

    request_routing = get_routing_for_object(_obj)
    request_routing.validate_metadata(params=kwargs, method=_method)
    routed_params = request_routing.route_params(params=kwargs, caller=_method)

    return routed_params

