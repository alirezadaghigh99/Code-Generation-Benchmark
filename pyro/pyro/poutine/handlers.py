def trace(
    fn: None = ...,
    graph_type: Optional[Literal["flat", "dense"]] = None,
    param_only: Optional[bool] = None,
) -> TraceMessenger: ...

def block(
    fn: None = ...,
    hide_fn: Optional[Callable[["Message"], Optional[bool]]] = None,
    expose_fn: Optional[Callable[["Message"], Optional[bool]]] = None,
    hide_all: bool = True,
    expose_all: bool = False,
    hide: Optional[List[str]] = None,
    expose: Optional[List[str]] = None,
    hide_types: Optional[List[str]] = None,
    expose_types: Optional[List[str]] = None,
) -> BlockMessenger: ...

def queue(
    fn=None,
    queue=None,
    max_tries=None,
    extend_fn=None,
    escape_fn=None,
    num_samples=None,
):
    """
    Used in sequential enumeration over discrete variables.

    Given a stochastic function and a queue,
    return a return value from a complete trace in the queue.

    :param fn: a stochastic function (callable containing Pyro primitive calls)
    :param queue: a queue data structure like multiprocessing.Queue to hold partial traces
    :param max_tries: maximum number of attempts to compute a single complete trace
    :param extend_fn: function (possibly stochastic) that takes a partial trace and a site,
        and returns a list of extended traces
    :param escape_fn: function (possibly stochastic) that takes a partial trace and a site,
        and returns a boolean value to decide whether to exit
    :param num_samples: optional number of extended traces for extend_fn to return
    :returns: stochastic function decorated with poutine logic
    """

    if max_tries is None:
        max_tries = int(1e6)

    if extend_fn is None:
        extend_fn = util.enum_extend

    if escape_fn is None:
        escape_fn = util.discrete_escape

    if num_samples is None:
        num_samples = -1

    def wrapper(wrapped):
        def _fn(*args, **kwargs):
            for i in range(max_tries):
                assert (
                    not queue.empty()
                ), "trying to get() from an empty queue will deadlock"

                next_trace = queue.get()
                try:
                    ftr = trace(
                        escape(
                            replay(wrapped, trace=next_trace),  # noqa: F821
                            escape_fn=functools.partial(escape_fn, next_trace),
                        )
                    )
                    return ftr(*args, **kwargs)
                except NonlocalExit as site_container:
                    site_container.reset_stack()
                    for tr in extend_fn(
                        ftr.trace.copy(), site_container.site, num_samples=num_samples
                    ):
                        queue.put(tr)

            raise ValueError("max tries ({}) exceeded".format(str(max_tries)))

        return _fn

    return wrapper(fn) if fn is not None else wrapper

