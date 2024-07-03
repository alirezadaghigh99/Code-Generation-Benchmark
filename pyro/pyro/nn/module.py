class PyroParam(NamedTuple):
    """
    Declares a Pyro-managed learnable attribute of a :class:`PyroModule`,
    similar to :func:`pyro.param <pyro.primitives.param>`.

    This can be used either to set attributes of :class:`PyroModule`
    instances::

        assert isinstance(my_module, PyroModule)
        my_module.x = PyroParam(torch.zeros(4))                   # eager
        my_module.y = PyroParam(lambda: torch.randn(4))           # lazy
        my_module.z = PyroParam(torch.ones(4),                    # eager
                                constraint=constraints.positive,
                                event_dim=1)

    or EXPERIMENTALLY as a decorator on lazy initialization properties::

        class MyModule(PyroModule):
            @PyroParam
            def x(self):
                return torch.zeros(4)

            @PyroParam
            def y(self):
                return torch.randn(4)

            @PyroParam(constraint=constraints.real, event_dim=1)
            def z(self):
                return torch.ones(4)

            def forward(self):
                return self.x + self.y + self.z  # accessed like a @property

    :param init_value: Either a tensor for eager initialization, a callable for
        lazy initialization, or None for use as a decorator.
    :type init_value: torch.Tensor or callable returning a torch.Tensor or None
    :param constraint: torch constraint, defaults to ``constraints.real``.
    :type constraint: ~torch.distributions.constraints.Constraint
    :param int event_dim: (optional) number of rightmost dimensions unrelated
        to baching. Dimension to the left of this will be considered batch
        dimensions; if the param statement is inside a subsampled plate, then
        corresponding batch dimensions of the parameter will be correspondingly
        subsampled. If unspecified, all dimensions will be considered event
        dims and no subsampling will be performed.
    """

    init_value: Optional[Union[torch.Tensor, Callable[[], torch.Tensor]]] = None
    constraint: constraints.Constraint = constraints.real
    event_dim: Optional[int] = None

    # Support use as a decorator.
    def __get__(
        self, obj: Optional["PyroModule"], obj_type: Type["PyroModule"]
    ) -> "PyroParam":
        assert issubclass(obj_type, PyroModule)
        if obj is None:
            return self

        name = self.init_value.__name__  # type: ignore[union-attr]
        if name not in obj.__dict__["_pyro_params"]:
            init_value, constraint, event_dim = self
            # bind method's self arg
            init_value = functools.partial(init_value, obj)  # type: ignore[arg-type]
            setattr(obj, name, PyroParam(init_value, constraint, event_dim))
        value: PyroParam = obj.__getattr__(name)
        return value

    # Support decoration with optional kwargs, e.g. @PyroParam(event_dim=0).
    def __call__(
        self, init_value: Union[torch.Tensor, Callable[[], torch.Tensor]]
    ) -> "PyroParam":
        assert self.init_value is None
        return PyroParam(init_value, self.constraint, self.event_dim)