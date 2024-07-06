def kron(*args):
    """Multi-argument kronecker product"""
    if len(args) == 1:
        return args[0]
    if len(args) == 2:
        return np.kron(args[0], args[1])
    return np.kron(args[0], kron(*args[1:]))

