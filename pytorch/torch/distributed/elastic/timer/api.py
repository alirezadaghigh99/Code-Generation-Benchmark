def expires(
    after: float, scope: Optional[str] = None, client: Optional[TimerClient] = None
):
    """
    Acquires a countdown timer that expires in ``after`` seconds from now,
    unless the code-block that it wraps is finished within the timeframe.
    When the timer expires, this worker is eligible to be reaped. The
    exact meaning of "reaped" depends on the client implementation. In
    most cases, reaping means to terminate the worker process.
    Note that the worker is NOT guaranteed to be reaped at exactly
    ``time.now() + after``, but rather the worker is "eligible" for being
    reaped and the ``TimerServer`` that the client talks to will ultimately
    make the decision when and how to reap the workers with expired timers.

    Usage::

        torch.distributed.elastic.timer.configure(LocalTimerClient())
        with expires(after=10):
            torch.distributed.all_reduce(...)
    """
    if client is None:
        if _timer_client is None:
            raise RuntimeError("Configure timer client before using countdown timers.")
        client = _timer_client
    if scope is None:
        # grab the caller file + lineno
        caller = getframeinfo(stack()[1][0])
        scope = f"{caller.filename}#{caller.lineno}"
    expiration = time.time() + after
    client.acquire(scope, expiration)
    try:
        yield
    finally:
        client.release(scope)

