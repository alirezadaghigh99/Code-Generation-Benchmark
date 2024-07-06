def ndiagquad(
    funcs: Union[Callable[..., tf.Tensor], Iterable[Callable[..., tf.Tensor]]],
    H: int,
    Fmu: Union[TensorType, Tuple[TensorType, ...], List[TensorType]],
    Fvar: Union[TensorType, Tuple[TensorType, ...], List[TensorType]],
    logspace: bool = False,
    **Ys: TensorType,
) -> tf.Tensor:
    """
    Computes N Gaussian expectation integrals of one or more functions
    using Gauss-Hermite quadrature. The Gaussians must be independent.

    The means and variances of the Gaussians are specified by Fmu and Fvar.
    The N-integrals are assumed to be taken wrt the last dimensions of Fmu, Fvar.

    `Fmu`, `Fvar`, `Ys` should all have same shape, with overall size `N`.

    :param funcs: the integrand(s):
        Callable or Iterable of Callables that operates elementwise
    :param H: number of Gauss-Hermite quadrature points
    :param Fmu: array/tensor or `Din`-tuple/list thereof
    :param Fvar: array/tensor or `Din`-tuple/list thereof
    :param logspace: if True, funcs are the log-integrands and this calculates
        the log-expectation of exp(funcs)
    :param Ys: arrays/tensors; deterministic arguments to be passed by name
    :return: shape is the same as that of the first Fmu
    """
    warnings.warn(
        "Please use gpflow.quadrature.NDiagGHQuadrature instead "
        "(note the changed convention of how multi-dimensional quadrature is handled)",
        DeprecationWarning,
    )

    n_gh = H
    if isinstance(Fmu, (tuple, list)):
        dim = len(Fmu)
        shape = tf.shape(Fmu[0])
        Fmu = tf.stack(Fmu, axis=-1)
        Fvar = tf.stack(Fvar, axis=-1)
    else:
        dim = 1
        shape = tf.shape(Fmu)

    Fmu = tf.reshape(Fmu, (-1, dim))
    Fvar = tf.reshape(Fvar, (-1, dim))

    Ys = {Yname: tf.reshape(Y, (-1, 1)) for Yname, Y in Ys.items()}

    def wrapper(old_fun: Callable[..., tf.Tensor]) -> Callable[..., tf.Tensor]:
        @wraps(old_fun)
        def new_fun(X: TensorType, **Ys: TensorType) -> tf.Tensor:
            Xs = tf.unstack(tf.expand_dims(X, axis=-2), axis=-1)
            fun_eval = old_fun(*Xs, **Ys)
            return tf.cond(
                pred=tf.less(tf.rank(fun_eval), tf.rank(X)),
                true_fn=lambda: fun_eval[..., tf.newaxis],
                false_fn=lambda: fun_eval,
            )

        return new_fun

    if isinstance(funcs, Iterable):
        funcs = [wrapper(f) for f in funcs]
    else:
        funcs = wrapper(funcs)

    quadrature = NDiagGHQuadrature(dim, n_gh)
    if logspace:
        result = quadrature.logspace(funcs, Fmu, Fvar, **Ys)
    else:
        result = quadrature(funcs, Fmu, Fvar, **Ys)

    if isinstance(result, list):
        result = [tf.reshape(r, shape) for r in result]
    else:
        result = tf.reshape(result, shape)

    return result

