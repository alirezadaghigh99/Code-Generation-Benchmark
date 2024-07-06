def mean(a, axis=None, keepdims: bool = False):
    """
    Compute the arithmetic mean along the specified axis and over MPI processes.

    Returns the average of the array elements. The average is taken over the flattened
    array by default, otherwise over the specified axis. float64 intermediate and
    return values are used for integer inputs.

    Args:
        a: The input array
        axis: Axis or axes along which the means are computed. The default (None) is
            to compute the mean of the flattened array.
        keepdims: If True the output array will have the same number of dimensions as
            the input, with the reduced axes having length 1. (default=False)

    Returns:
        The array with reduced dimensions defined by axis.

    """
    out = a.mean(axis=axis, keepdims=keepdims)

    out, _ = mpi.mpi_mean_jax(out)
    return out

def subtract_mean(x, axis=None):
    """
    Subtracts the mean of the input array over all but the last dimension
    and over all MPI processes from each entry.

    Args:
        x: Input array
        axis: Axis or axes along which the means are computed. The default (None) is to
              compute the mean of the flattened array.

    Returns:
        The resulting array.

    """
    # here we keep the dims, since automatic broadcasting of a scalar (shape () )
    # to an array produces errors when used inside of a function which is transposed
    # with jax.linear_transpose
    x_mean = mean(x, axis=axis, keepdims=True)
    return x - x_mean  # automatic broadcasting of x_mean

