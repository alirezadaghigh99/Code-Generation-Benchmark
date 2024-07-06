def leading_transpose(tensor: tf.Tensor, perm: List[Any], leading_dim: int = 0) -> tf.Tensor:
    """
    Transposes tensors with leading dimensions.

    Leading dimensions in permutation list represented via ellipsis `...` and is of type
    List[Union[int, type(...)]  (please note, due to mypy issues, List[Any] is used instead).  When
    leading dimensions are found, `transpose` method considers them as a single grouped element
    indexed by 0 in `perm` list. So, passing `perm=[-2, ..., -1]`, you assume that your input tensor
    has [..., A, B] shape, and you want to move leading dims between A and B dimensions.  Dimension
    indices in permutation list can be negative or positive. Valid positive indices start from 1 up
    to the tensor rank, viewing leading dimensions `...` as zero index.

    Example::

        a = tf.random.normal((1, 2, 3, 4, 5, 6))
        # [..., A, B, C],
        # where A is 1st element,
        # B is 2nd element and
        # C is 3rd element in
        # permutation list,
        # leading dimensions are [1, 2, 3]
        # which are 0th element in permutation list
        b = leading_transpose(a, [3, -3, ..., -2])  # [C, A, ..., B]
        sess.run(b).shape

        output> (6, 4, 1, 2, 3, 5)

    :param tensor: TensorFlow tensor.
    :param perm: List of permutation indices.
    :returns: TensorFlow tensor.
    :raises ValueError: when `...` cannot be found.

    """
    perm = copy.copy(perm)
    idx = perm.index(...)
    perm[idx] = leading_dim

    rank = tf.rank(tensor)
    perm_tf = perm % rank

    leading_dims = tf.range(rank - len(perm) + 1)
    perm = tf.concat([perm_tf[:idx], leading_dims, perm_tf[idx + 1 :]], 0)
    return tf.transpose(tensor, perm)

def difference_matrix(X: tf.Tensor, X2: Optional[tf.Tensor]) -> tf.Tensor:
    """
    Returns (X - X2ᵀ)
    """
    if X2 is None:
        X2 = X
        diff = X[..., :, tf.newaxis, :] - X2[..., tf.newaxis, :, :]
        return diff
    Xshape = tf.shape(X)
    X2shape = tf.shape(X2)
    X = tf.reshape(X, (-1, Xshape[-1]))
    X2 = tf.reshape(X2, (-1, X2shape[-1]))
    diff = X[:, tf.newaxis, :] - X2[tf.newaxis, :, :]
    diff = tf.reshape(diff, tf.concat((Xshape[:-1], X2shape[:-1], [Xshape[-1]]), 0))
    return diff

