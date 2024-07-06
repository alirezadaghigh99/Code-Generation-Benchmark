def array(*args, like=None, **kwargs):
    """Creates an array or tensor object of the target framework.

    If the PyTorch interface is specified, this method preserves the Torch device used.
    If the JAX interface is specified, this method uses JAX numpy arrays, which do not cause issues with jit tracers.

    Returns:
        tensor_like: the tensor_like object of the framework
    """
    res = np.array(*args, like=like, **kwargs)
    if like is not None and get_interface(like) == "torch":
        res = res.to(device=like.device)
    return res

def dot(tensor1, tensor2, like=None):
    """Returns the matrix or dot product of two tensors.

    * If either tensor is 0-dimensional, elementwise multiplication
      is performed and a 0-dimensional scalar or a tensor with the
      same dimensions as the other tensor is returned.

    * If both tensors are 1-dimensional, the dot product is returned.

    * If the first array is 2-dimensional and the second array 1-dimensional,
      the matrix-vector product is returned.

    * If both tensors are 2-dimensional, the matrix product is returned.

    * Finally, if the first array is N-dimensional and the second array
      M-dimensional, a sum product over the last dimension of the first array,
      and the second-to-last dimension of the second array is returned.

    Args:
        tensor1 (tensor_like): input tensor
        tensor2 (tensor_like): input tensor

    Returns:
        tensor_like: the matrix or dot product of two tensors
    """
    x, y = np.coerce([tensor1, tensor2], like=like)

    if like == "torch":

        if x.ndim == 0 or y.ndim == 0:
            return x * y

        if x.ndim <= 2 and y.ndim <= 2:
            return x @ y

        return np.tensordot(x, y, axes=[[-1], [-2]], like=like)

    if like in {"tensorflow", "autograd"}:

        ndim_y = len(np.shape(y))
        ndim_x = len(np.shape(x))

        if ndim_x == 0 or ndim_y == 0:
            return x * y

        if ndim_y == 1:
            return np.tensordot(x, y, axes=[[-1], [0]], like=like)

        if ndim_x == 2 and ndim_y == 2:
            return x @ y

        return np.tensordot(x, y, axes=[[-1], [-2]], like=like)

    return np.dot(x, y, like=like)

def stack(values, axis=0, like=None):
    """Stack a sequence of tensors along the specified axis.

    .. warning::

        Tensors that are incompatible (such as Torch and TensorFlow tensors)
        cannot both be present.

    Args:
        values (Sequence[tensor_like]): Sequence of tensor-like objects to
            stack. Each object in the sequence must have the same size in the given axis.
        axis (int): The axis along which the input tensors are stacked. ``axis=0`` corresponds
            to vertical stacking.

    Returns:
        tensor_like: The stacked array. The stacked array will have one additional dimension
        compared to the unstacked tensors.

    **Example**

    >>> x = tf.constant([0.6, 0.1, 0.6])
    >>> y = tf.Variable([0.1, 0.2, 0.3])
    >>> z = np.array([5., 8., 101.])
    >>> stack([x, y, z])
    <tf.Tensor: shape=(3, 3), dtype=float32, numpy=
    array([[6.00e-01, 1.00e-01, 6.00e-01],
           [1.00e-01, 2.00e-01, 3.00e-01],
           [5.00e+00, 8.00e+00, 1.01e+02]], dtype=float32)>
    """
    values = np.coerce(values, like=like)
    return np.stack(values, axis=axis, like=like)

def eye(*args, like=None, **kwargs):
    """Creates an identity array or tensor object of the target framework.

    This method preserves the Torch device used.

    Returns:
        tensor_like: the tensor_like object of the framework
    """
    res = np.eye(*args, like=like, **kwargs)
    if like is not None and get_interface(like) == "torch":
        res = res.to(device=like.device)
    return res

def add(*args, like=None, **kwargs):
    """Add arguments element-wise."""
    if like == "scipy":
        return onp.add(*args, **kwargs)  # Dispatch scipy add to numpy backed specifically.

    arg_interfaces = {get_interface(args[0]), get_interface(args[1])}

    # case of one torch tensor and one vanilla numpy array
    if like == "torch" and len(arg_interfaces) == 2:
        # In autoray 0.6.5, np.add dispatches to torch instead of
        # numpy if one parameter is a torch tensor and the other is
        # a numpy array. torch.add raises an Exception if one of the
        # arguments is a numpy array, so here we cast both arguments
        # to be tensors.
        dev = getattr(args[0], "device", None) or getattr(args[1], "device")
        arg0 = np.asarray(args[0], device=dev, like=like)
        arg1 = np.asarray(args[1], device=dev, like=like)
        return np.add(arg0, arg1, *args[2:], **kwargs)

    return np.add(*args, **kwargs, like=like)

def norm(tensor, like=None, **kwargs):
    """Compute the norm of a tensor in each interface."""
    if like == "jax":
        from jax.numpy.linalg import norm

    elif like == "tensorflow":
        from tensorflow import norm

    elif like == "torch":
        from torch.linalg import norm

        if "axis" in kwargs:
            axis_val = kwargs.pop("axis")
            kwargs["dim"] = axis_val

    elif (
        like == "autograd" and kwargs.get("ord", None) is None and kwargs.get("axis", None) is None
    ):
        norm = _flat_autograd_norm

    else:
        from scipy.linalg import norm

    return norm(tensor, **kwargs)

def concatenate(values, axis=0, like=None):
    """Concatenate a sequence of tensors along the specified axis.

    .. warning::

        Tensors that are incompatible (such as Torch and TensorFlow tensors)
        cannot both be present.

    Args:
        values (Sequence[tensor_like]): Sequence of tensor-like objects to
            concatenate. The objects must have the same shape, except in the dimension corresponding
            to axis (the first, by default).
        axis (int): The axis along which the input tensors are concatenated. If axis is None,
            tensors are flattened before use. Default is 0.

    Returns:
        tensor_like: The concatenated tensor.

    **Example**

    >>> x = tf.constant([0.6, 0.1, 0.6])
    >>> y = tf.Variable([0.1, 0.2, 0.3])
    >>> z = np.array([5., 8., 101.])
    >>> concatenate([x, y, z])
    <tf.Tensor: shape=(9,), dtype=float32, numpy=
    array([6.00e-01, 1.00e-01, 6.00e-01, 1.00e-01, 2.00e-01, 3.00e-01,
           5.00e+00, 8.00e+00, 1.01e+02], dtype=float32)>
    """

    if like == "torch":
        import torch

        device = (
            "cuda"
            if any(t.device.type == "cuda" for t in values if isinstance(t, torch.Tensor))
            else "cpu"
        )

        if axis is None:
            # flatten and then concatenate zero'th dimension
            # to reproduce numpy's behaviour
            values = [
                np.flatten(torch.as_tensor(t, device=torch.device(device)))  # pragma: no cover
                for t in values
            ]
            axis = 0
        else:
            values = [
                torch.as_tensor(t, device=torch.device(device)) for t in values  # pragma: no cover
            ]

    if like == "tensorflow" and axis is None:
        # flatten and then concatenate zero'th dimension
        # to reproduce numpy's behaviour
        values = [np.flatten(np.array(t)) for t in values]
        axis = 0

    return np.concatenate(values, axis=axis, like=like)

