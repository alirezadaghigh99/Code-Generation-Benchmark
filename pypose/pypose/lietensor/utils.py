def randn_SE3(*lsize, sigma=1.0, **kwargs):
    r'''
    Returns :obj:`SE3_type` LieTensor filled with the Exponential map of the random
    :obj:`se3_type` LieTensor generated using :meth:`pypose.randn_se3()`.

    .. math::
        \mathrm{data}[*, :] = \mathrm{Exp}([\tau_x, \tau_y, \tau_z, \delta_x, \delta_y, \delta_z]),

    where :math:`[\tau_x, \tau_y, \tau_z]` is generated from a normal distribution
    :math:`\mathcal{N}(0, \sigma_t)`, :math:`[\delta_x, \delta_y, \delta_z]` is
    generated using :meth:`pypose.randn_so3()` with with standard deviation
    :math:`\sigma_r`, and :math:`\mathrm{Exp}()` is the Exponential map. Note that standard
    deviations :math:`\sigma_t` and :math:`\sigma_r` are specified by ``sigma`` (:math:`\sigma`),
    where :math:`\sigma = (\sigma_t, \sigma_r)`.

    For detailed explanation, please see :meth:`pypose.randn_se3()` and :meth:`pypose.Exp()`.

    Args:
        lsize (int...): a sequence of integers defining the lshape of the output tensor.
            Can be a variable number of arguments or a collection like a list or tuple.

        sigma (float or (float...), optional): standard deviation
            (:math:`\sigma_t` and :math:`\sigma_r`) for the two normal distribution.
            Default: ``1.0``.

        requires_grad (bool, optional): If autograd should record operations on
            the returned tensor. Default: ``False``.

        generator (torch.Generator, optional): a pseudorandom number generator for sampling

        dtype (torch.dtype, optional): the desired data type of returned tensor.
            Default: ``None``. If None, uses a global default
            (see :meth:`torch.set_default_tensor_type()`).

        layout (torch.layout, optional): the desired layout of returned Tensor.
            Default: ``torch.strided``.

        device (torch.device, optional): the desired device of returned tensor.
            Default: ``None``. If ``None``, uses the current device for the default tensor
            type (see :meth:`torch.set_default_tensor_type()`). Device will be the CPU
            for CPU tensor types and the current CUDA device for CUDA tensor types.

    Returns:
        LieTensor: a :obj:`SE3_type` LieTensor

    Note:
        The parameter :math:`\sigma` can either be:

        - a single ``float`` -- in which all the elements in the :obj:`SE3_type`
          share the same sigma, i.e.,
          :math:`\sigma_{\rm{t}}` = :math:`\sigma_{\rm{r}}` = :math:`\sigma`.
        - a ``tuple`` of two floats -- in which case, the specific sigmas are
          assigned independently, i.e.,
          :math:`\sigma` = (:math:`\sigma_{\rm{t}}`, :math:`\sigma_{\rm{r}}`).
        - a ``tuple`` of four floats -- in which case, the specific sigmas for
          each translation data are assigned independently, i.e.,
          :math:`\sigma` = (:math:`\sigma_{\rm{tx}}`, :math:`\sigma_{\rm{ty}}`,
          :math:`\sigma_{\rm{tz}}`, :math:`\sigma_{\rm{r}}`).

    Example:

        For :math:`\sigma = (\sigma_t, \sigma_r)`

        >>> pp.randn_SE3(2, sigma=(1.0, 2.0))
        SE3Type LieTensor:
        tensor([[ 0.2947, -1.6990, -0.5535,  0.4439,  0.2777,  0.0518,  0.8504],
                [ 0.6825,  0.2963,  0.3410,  0.3375, -0.2355,  0.7389, -0.5335]])

        For :math:`\sigma = (\sigma_{tx}, \sigma_{ty}, \sigma_{tz}, \sigma_{r})`

        >>> pp.randn_SE3(2, sigma=(1.0, 1.5, 2.0, 2.0))
        SE3Type LieTensor:
        tensor([[-1.5689, -0.6772,  0.3580, -0.2509,  0.8257, -0.4950,  0.1018],
                [ 0.2613, -2.7613,  0.2151, -0.8802,  0.2619,  0.3044,  0.2531]])
    '''
    return SE3_type.randn(*lsize, sigma=sigma, **kwargs)

