def make_boxed_func(f):
    def g(args):
        return f(*args)

    g._boxed_call = True  # type: ignore[attr-defined]
    return g

