class Parameter(torch.Tensor, metaclass=_ParameterMeta):
    r"""A kind of Tensor that is to be considered a module parameter.

    Parameters are :class:`~torch.Tensor` subclasses, that have a
    very special property when used with :class:`Module` s - when they're
    assigned as Module attributes they are automatically added to the list of
    its parameters, and will appear e.g. in :meth:`~Module.parameters` iterator.
    Assigning a Tensor doesn't have such effect. This is because one might
    want to cache some temporary state, like last hidden state of the RNN, in
    the model. If there was no such class as :class:`Parameter`, these
    temporaries would get registered too.

    Args:
        data (Tensor): parameter tensor.
        requires_grad (bool, optional): if the parameter requires gradient. Note that
            the torch.no_grad() context does NOT affect the default behavior of
            Parameter creation--the Parameter will still have `requires_grad=True` in
            :class:`~no_grad` mode. See :ref:`locally-disable-grad-doc` for more
            details. Default: `True`
    """

    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            data = torch.empty(0)
        if type(data) is torch.Tensor or type(data) is Parameter:
            # For ease of BC maintenance, keep this path for standard Tensor.
            # Eventually (tm), we should change the behavior for standard Tensor to match.
            return torch.Tensor._make_subclass(cls, data, requires_grad)

        # Path for custom tensors: set a flag on the instance to indicate parameter-ness.
        t = data.detach().requires_grad_(requires_grad)
        if type(t) is not type(data):
            raise RuntimeError(
                f"Creating a Parameter from an instance of type {type(data).__name__} "
                "requires that detach() returns an instance of the same type, but return "
                f"type {type(t).__name__} was found instead. To use the type as a "
                "Parameter, please correct the detach() semantics defined by "
                "its __torch_dispatch__() implementation."
            )
        t._is_param = True
        return t

    # Note: the 3 methods below only apply to standard Tensor. Parameters of custom tensor types
    # are still considered that custom tensor type and these methods will not be called for them.
    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(
                self.data.clone(memory_format=torch.preserve_format), self.requires_grad
            )
            memo[id(self)] = result
            return result

    def __repr__(self):
        return "Parameter containing:\n" + super().__repr__()

    def __reduce_ex__(self, proto):
        state = torch._utils._get_obj_state(self)

        # See Note [Don't serialize hooks]
        hooks = OrderedDict()
        if not state:
            return (
                torch._utils._rebuild_parameter,
                (self.data, self.requires_grad, hooks),
            )

        return (
            torch._utils._rebuild_parameter_with_state,
            (self.data, self.requires_grad, hooks, state),
        )

    __torch_function__ = _disabled_torch_function_impl

class UninitializedParameter(UninitializedTensorMixin, Parameter):
    r"""A parameter that is not initialized.

    Uninitialized Parameters are a a special case of :class:`torch.nn.Parameter`
    where the shape of the data is still unknown.

    Unlike a :class:`torch.nn.Parameter`, uninitialized parameters
    hold no data and attempting to access some properties, like their shape,
    will throw a runtime error. The only operations that can be performed on a uninitialized
    parameter are changing its datatype, moving it to a different device and
    converting it to a regular :class:`torch.nn.Parameter`.

    The default device or dtype to use when the parameter is materialized can be set
    during construction using e.g. ``device='cuda'``.
    """

    cls_to_become = Parameter

    def __new__(cls, requires_grad=True, device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        data = torch.empty(0, **factory_kwargs)
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(self.requires_grad, self.data.device, self.data.dtype)
            memo[id(self)] = result
            return result

class UninitializedBuffer(UninitializedTensorMixin, torch.Tensor):
    r"""A buffer that is not initialized.

    Uninitialized Buffer is a a special case of :class:`torch.Tensor`
    where the shape of the data is still unknown.

    Unlike a :class:`torch.Tensor`, uninitialized parameters
    hold no data and attempting to access some properties, like their shape,
    will throw a runtime error. The only operations that can be performed on a uninitialized
    parameter are changing its datatype, moving it to a different device and
    converting it to a regular :class:`torch.Tensor`.

    The default device or dtype to use when the buffer is materialized can be set
    during construction using e.g. ``device='cuda'``.
    """

    cls_to_become = torch.Tensor

    def __new__(cls, requires_grad=False, device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        data = torch.empty(0, **factory_kwargs)
        return torch.Tensor._make_subclass(cls, data, requires_grad)

