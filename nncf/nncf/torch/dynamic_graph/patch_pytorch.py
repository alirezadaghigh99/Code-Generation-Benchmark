def register_operator(name=None):
    def wrap(operator):
        op_name = name
        if op_name is None:
            op_name = operator.__name__
        return wrap_operator(operator, PatchedOperatorInfo(op_name, NamespaceTarget.EXTERNAL))

    return wrap

