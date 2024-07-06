def resample(
    vol: tf.Tensor,
    loc: tf.Tensor,
    interpolation: str = "linear",
    zero_boundary: bool = True,
) -> tf.Tensor:
    r"""
    Sample the volume at given locations.

    Input has

    - volume, vol, of shape = (batch, v_dim 1, ..., v_dim n),
      or (batch, v_dim 1, ..., v_dim n, ch),
      where n is the dimension of volume,
      ch is the extra dimension as features.

      Denote vol_shape = (v_dim 1, ..., v_dim n)

    - location, loc, of shape = (batch, l_dim 1, ..., l_dim m, n),
      where m is the dimension of output.

      Denote loc_shape = (l_dim 1, ..., l_dim m)

    Reference:

    - neuron's interpn
      https://github.com/adalca/neurite/blob/legacy/neuron/utils.py

      Difference

      1. they dont have batch size
      2. they support more dimensions in vol

      TODO try not using stack as neuron claims it's slower

    :param vol: shape = (batch, \*vol_shape) or (batch, \*vol_shape, ch)
      with the last channel for features
    :param loc: shape = (batch, \*loc_shape, n)
      such that loc[b, l1, ..., lm, :] = [v1, ..., vn] is of shape (n,),
      which represents a point in vol, with coordinates (v1, ..., vn)
    :param interpolation: linear only, TODO support nearest
    :param zero_boundary: if true, values on or outside boundary will be zeros
    :return: shape = (batch, \*loc_shape) or (batch, \*loc_shape, ch)
    """

    if interpolation != "linear":
        raise ValueError("resample supports only linear interpolation")

    # init
    batch_size = vol.shape[0]
    loc_shape = loc.shape[1:-1]
    dim_vol = loc.shape[-1]  # dimension of vol, n
    if dim_vol == len(vol.shape) - 1:
        # vol.shape = (batch, *vol_shape)
        has_ch = False
    elif dim_vol == len(vol.shape) - 2:
        # vol.shape = (batch, *vol_shape, ch)
        has_ch = True
    else:
        raise ValueError(
            "vol shape inconsistent with loc "
            "vol.shape = {}, loc.shape = {}".format(vol.shape, loc.shape)
        )
    vol_shape = vol.shape[1 : dim_vol + 1]

    # get floor/ceil for loc and stack, then clip together
    # loc, loc_floor, loc_ceil are have shape (batch, *loc_shape, n)
    loc_ceil = tf.math.ceil(loc)
    loc_floor = loc_ceil - 1
    # (batch, *loc_shape, n, 3)
    clipped = tf.stack([loc, loc_floor, loc_ceil], axis=-1)
    clip_value_max = tf.cast(vol_shape, dtype=clipped.dtype) - 1  # (n,)
    clipped_shape = [1] * (len(loc_shape) + 1) + [dim_vol, 1]
    clip_value_max = tf.reshape(clip_value_max, shape=clipped_shape)
    clipped = tf.clip_by_value(clipped, clip_value_min=0, clip_value_max=clip_value_max)

    # loc_floor_ceil has n sublists
    # each one corresponds to the floor and ceil coordinates for d-th dimension
    # each tensor is of shape (batch, *loc_shape), dtype int32

    # weight_floor has n tensors
    # each tensor is the weight for the corner of floor coordinates
    # each tensor's shape is (batch, *loc_shape) if volume has no feature channel
    #                        (batch, *loc_shape, 1) if volume has feature channel
    loc_floor_ceil, weight_floor, weight_ceil = [], [], []
    # using for loop is faster than using list comprehension
    for dim in range(dim_vol):
        # shape = (batch, *loc_shape)
        c_clipped = clipped[..., dim, 0]
        c_floor = clipped[..., dim, 1]
        c_ceil = clipped[..., dim, 2]
        w_floor = c_ceil - c_clipped  # shape = (batch, *loc_shape)
        w_ceil = c_clipped - c_floor if zero_boundary else 1 - w_floor
        if has_ch:
            w_floor = tf.expand_dims(w_floor, -1)  # shape = (batch, *loc_shape, 1)
            w_ceil = tf.expand_dims(w_ceil, -1)  # shape = (batch, *loc_shape, 1)

        loc_floor_ceil.append([tf.cast(c_floor, tf.int32), tf.cast(c_ceil, tf.int32)])
        weight_floor.append(w_floor)
        weight_ceil.append(w_ceil)

    # 2**n corners, each is a list of n binary values
    corner_indices = get_n_bits_combinations(num_bits=len(vol_shape))

    # batch_coords[b, l1, ..., lm] = b
    # range(batch_size) on axis 0 and repeated on other axes
    # add batch coords manually is faster than using batch_dims in tf.gather_nd
    batch_coords = tf.tile(
        tf.reshape(tf.range(batch_size), [batch_size] + [1] * len(loc_shape)),
        [1] + loc_shape,
    )  # shape = (batch, *loc_shape)

    # get vol values on n-dim hypercube corners
    # corner_values has 2 ** n elements
    # each of shape (batch, *loc_shape) or (batch, *loc_shape, ch)
    corner_values = [
        tf.gather_nd(
            vol,  # shape = (batch, *vol_shape) or (batch, *vol_shape, ch)
            tf.stack(
                [batch_coords]
                + [loc_floor_ceil[axis][fc_idx] for axis, fc_idx in enumerate(c)],
                axis=-1,
            ),  # shape = (batch, *loc_shape, n+1) after stack
        )
        for c in corner_indices  # c is list of len n
    ]  # each tensor has shape (batch, *loc_shape) or (batch, *loc_shape, ch)

    # resample
    sampled = pyramid_combination(
        values=corner_values, weight_floor=weight_floor, weight_ceil=weight_ceil
    )
    return sampled

def pyramid_combination(
    values: list, weight_floor: list, weight_ceil: list
) -> tf.Tensor:
    r"""
    Calculates linear interpolation (a weighted sum) using values of
    hypercube corners in dimension n.

    For example, when num_dimension = len(loc_shape) = num_bits = 3
    values correspond to values at corners of following coordinates

    .. code-block:: python

        [[0, 0, 0], # even
         [0, 0, 1], # odd
         [0, 1, 0], # even
         [0, 1, 1], # odd
         [1, 0, 0], # even
         [1, 0, 1], # odd
         [1, 1, 0], # even
         [1, 1, 1]] # odd

    values[::2] correspond to the corners with last coordinate == 0

    .. code-block:: python

        [[0, 0, 0],
         [0, 1, 0],
         [1, 0, 0],
         [1, 1, 0]]

    values[1::2] correspond to the corners with last coordinate == 1

    .. code-block:: python

        [[0, 0, 1],
         [0, 1, 1],
         [1, 0, 1],
         [1, 1, 1]]

    The weights correspond to the floor corners.
    For example, when num_dimension = len(loc_shape) = num_bits = 3,
    weight_floor = [f1, f2, f3] (ignoring the batch dimension).
    weight_ceil = [c1, c2, c3] (ignoring the batch dimension).

    So for corner with coords (x, y, z), x, y, z's values are 0 or 1

    - weight for x = f1 if x = 0 else c1
    - weight for y = f2 if y = 0 else c2
    - weight for z = f3 if z = 0 else c3

    so the weight for (x, y, z) is

    .. code-block:: text

        W_xyz = ((1-x) * f1 + x * c1)
              * ((1-y) * f2 + y * c2)
              * ((1-z) * f3 + z * c3)

    Let

    .. code-block:: text

        W_xy = ((1-x) * f1 + x * c1)
             * ((1-y) * f2 + y * c2)

    Then

    - W_xy0 = W_xy * f3
    - W_xy1 = W_xy * c3

    Similar to W_xyz, denote V_xyz the value at (x, y, z),
    the final sum V equals

    .. code-block:: text

          sum over x,y,z (V_xyz * W_xyz)
        = sum over x,y (V_xy0 * W_xy0 + V_xy1 * W_xy1)
        = sum over x,y (V_xy0 * W_xy * f3 + V_xy1 * W_xy * c3)
        = sum over x,y (V_xy0 * W_xy) * f3 + sum over x,y (V_xy1 * W_xy) * c3

    That's why we call this pyramid combination.
    It calculates the linear interpolation gradually, starting from
    the last dimension.
    The key is that the weight of each corner is the product of the weights
    along each dimension.

    :param values: a list having values on the corner,
                   it has 2**n tensors of shape
                   (\*loc_shape) or (batch, \*loc_shape) or (batch, \*loc_shape, ch)
                   the order is consistent with get_n_bits_combinations
                   loc_shape is independent from n, aka num_dim
    :param weight_floor: a list having weights of floor points,
                    it has n tensors of shape
                    (\*loc_shape) or (batch, \*loc_shape) or (batch, \*loc_shape, 1)
    :param weight_ceil: a list having weights of ceil points,
                    it has n tensors of shape
                    (\*loc_shape) or (batch, \*loc_shape) or (batch, \*loc_shape, 1)
    :return: one tensor of the same shape as an element in values
             (\*loc_shape) or (batch, \*loc_shape) or (batch, \*loc_shape, 1)
    """
    v_shape = values[0].shape
    wf_shape = weight_floor[0].shape
    wc_shape = weight_ceil[0].shape
    if len(v_shape) != len(wf_shape) or len(v_shape) != len(wc_shape):
        raise ValueError(
            "In pyramid_combination, elements of "
            "values, weight_floor, and weight_ceil should have same dimension. "
            f"value shape = {v_shape}, "
            f"weight_floor = {wf_shape}, "
            f"weight_ceil = {wc_shape}."
        )
    if 2 ** len(weight_floor) != len(values):
        raise ValueError(
            "In pyramid_combination, "
            "num_dim = len(weight_floor), "
            "len(values) must be 2 ** num_dim, "
            f"But len(weight_floor) = {len(weight_floor)}, "
            f"len(values) = {len(values)}"
        )

    if len(weight_floor) == 1:  # one dimension
        return values[0] * weight_floor[0] + values[1] * weight_ceil[0]
    # multi dimension
    values_floor = pyramid_combination(
        values=values[::2],
        weight_floor=weight_floor[:-1],
        weight_ceil=weight_ceil[:-1],
    )
    values_floor = values_floor * weight_floor[-1]
    values_ceil = pyramid_combination(
        values=values[1::2],
        weight_floor=weight_floor[:-1],
        weight_ceil=weight_ceil[:-1],
    )
    values_ceil = values_ceil * weight_ceil[-1]
    return values_floor + values_ceil

def _deconv_output_padding(
    input_shape: int, output_shape: int, kernel_size: int, stride: int, padding: str
) -> int:
    """
    Calculate output padding for Conv3DTranspose in 1D.

    - output_shape = (input_shape - 1)*stride + kernel_size - 2*pad + output_padding
    - output_padding = output_shape - ((input_shape - 1)*stride + kernel_size - 2*pad)

    Reference:

    - https://github.com/tensorflow/tensorflow/blob/r2.3/tensorflow/python/keras/utils/conv_utils.py#L140

    :param input_shape: shape of Conv3DTranspose input tensor
    :param output_shape: shape of Conv3DTranspose output tensor
    :param kernel_size: kernel size of Conv3DTranspose layer
    :param stride: stride of Conv3DTranspose layer
    :param padding: padding of Conv3DTranspose layer
    :return: output_padding for Conv3DTranspose layer
    """
    if padding == "same":
        pad = kernel_size // 2
    elif padding == "valid":
        pad = 0
    elif padding == "full":
        pad = kernel_size - 1
    else:
        raise ValueError(f"Unknown padding {padding} in deconv_output_padding")
    return output_shape - ((input_shape - 1) * stride + kernel_size - 2 * pad)

def gaussian_filter_3d(kernel_sigma: Union[Tuple, List]) -> tf.Tensor:
    """
    Define a gaussian filter in 3d for smoothing.

    The filter size is defined 3*kernel_sigma


    :param kernel_sigma: the deviation at each direction (list)
        or use an isotropic deviation (int)
    :return: kernel: tf.Tensor specify a gaussian kernel of shape:
        [3*k for k in kernel_sigma]
    """
    if isinstance(kernel_sigma, (int, float)):
        kernel_sigma = (kernel_sigma, kernel_sigma, kernel_sigma)

    kernel_size = [
        int(np.ceil(ks * 3) + np.mod(np.ceil(ks * 3) + 1, 2)) for ks in kernel_sigma
    ]

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    coord = [np.arange(ks) for ks in kernel_size]

    xx, yy, zz = np.meshgrid(coord[0], coord[1], coord[2], indexing="ij")
    xyz_grid = np.concatenate(
        (xx[np.newaxis], yy[np.newaxis], zz[np.newaxis]), axis=0
    )  # 2, y, x

    mean = np.asarray([(ks - 1) / 2.0 for ks in kernel_size])
    mean = mean.reshape(-1, 1, 1, 1)
    variance = np.asarray([ks ** 2.0 for ks in kernel_sigma])
    variance = variance.reshape(-1, 1, 1, 1)

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    # 2.506628274631 = sqrt(2 * pi)

    norm_kernel = 1.0 / (np.sqrt(2 * np.pi) ** 3 + np.prod(kernel_sigma))
    kernel = norm_kernel * np.exp(
        -np.sum((xyz_grid - mean) ** 2.0 / (2 * variance), axis=0)
    )

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / np.sum(kernel)

    # Reshape
    kernel = kernel.reshape(kernel_size[0], kernel_size[1], kernel_size[2])

    # Total kernel
    total_kernel = np.zeros(tuple(kernel_size) + (3, 3))
    total_kernel[..., 0, 0] = kernel
    total_kernel[..., 1, 1] = kernel
    total_kernel[..., 2, 2] = kernel

    return tf.convert_to_tensor(total_kernel, dtype=tf.float32)

