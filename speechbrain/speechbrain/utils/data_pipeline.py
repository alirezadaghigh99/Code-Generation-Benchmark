def provides(*output_keys):
    """Decorator which makes a DynamicItem and specifies what keys it provides.

    If the wrapped object is a generator function (has a yield statement),
    Creates a GeneratorDynamicItem. If the object is already a DynamicItem,
    just specifies the provided keys for that. Otherwise creates a new regular
    DynamicItem, with provided keys specified.

    Arguments
    ---------
    *output_keys : tuple
        The data keys to be produced by this function

    Returns
    -------
    The decorated function, with output keys specified

    NOTE
    ----
    The behavior is slightly different for generators and regular functions, if
    many output keys are specified, e.g. @provides("signal", "mfcc"). Regular
    functions should return a tuple with len equal to len(output_keys), while
    generators should yield the items one by one.

    >>> @provides("signal", "feat")
    ... def read_feat():
    ...     wav = [.1,.2,-.1]
    ...     feat = [s**2 for s in wav]
    ...     return wav, feat
    >>> @provides("signal", "feat")
    ... def read_feat():
    ...     wav = [.1,.2,-.1]
    ...     yield wav
    ...     feat = [s**2 for s in wav]
    ...     yield feat

    If multiple keys are yielded at once, write e.g.,

    >>> @provides("wav_read", ["left_channel", "right_channel"])
    ... def read_multi_channel():
    ...     wav = [[.1,.2,-.1],[.2,.1,-.1]]
    ...     yield wav
    ...     yield wav[0], wav[1]

    """

    def decorator(obj):
        """Decorator definition."""
        if isinstance(obj, DynamicItem):
            if obj.provides:
                raise ValueError("Can't overwrite DynamicItem provides-list.")
            obj.provides = output_keys
            return obj
        elif inspect.isgeneratorfunction(obj):
            return GeneratorDynamicItem(func=obj, provides=output_keys)
        else:
            return DynamicItem(func=obj, provides=output_keys)

    return decorator

