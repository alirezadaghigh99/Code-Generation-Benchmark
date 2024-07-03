def create_variable(value, name, shape, trainable=True):
    """
    Creates NN parameter as Tensorfow variable.

    Parameters
    ----------
    value : array-like, Tensorfow variable, scalar or Initializer
        Default value for the parameter.

    name : str
        Shared variable name.

    shape : tuple
        Parameter's shape.

    trainable : bool
        Whether parameter trainable by backpropagation.

    Returns
    -------
    Tensorfow variable.
    """
    from neupy import init

    if shape is not None:
        shape = shape_to_tuple(shape)

    if isinstance(value, (tf.Variable, tf.Tensor, np.ndarray, np.matrix)):
        variable_shape = shape_to_tuple(value.shape)

        if as_tuple(variable_shape) != as_tuple(shape):
            raise ValueError(
                "Cannot create variable with name `{}`. Provided variable "
                "with shape {} is incompatible with expected shape {}"
                "".format(name, variable_shape, shape))

    if isinstance(value, (tf.Variable, tf.Tensor)):
        return value

    if isinstance(value, (int, float)):
        value = init.Constant(value)

    if isinstance(value, init.Initializer):
        value = value.sample(shape)

    return tf.Variable(
        asfloat(value),
        name=name,
        dtype=tf.float32,
        trainable=trainable,
    )