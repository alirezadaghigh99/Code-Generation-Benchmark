def require_backward(tensor):
    """
    Marks a tensor as a leaf in the adjoint graph.
    """
    tensor._pyro_backward = _LeafBackward(tensor)

