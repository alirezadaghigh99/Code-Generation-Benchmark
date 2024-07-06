def get_default_args(C, *, _do_not_process: Tuple[type, ...] = ()) -> DictConfig:
    """
    Get the DictConfig corresponding to the defaults in a dataclass or
    configurable. Normal use is to provide a dataclass can be provided as C.
    If enable_get_default_args has been called on a function or plain class,
    then that function or class can be provided as C.

    If C is a subclass of Configurable or ReplaceableBase, we make sure
    it has been processed with expand_args_fields.

    Args:
        C: the class or function to be processed
        _do_not_process: (internal use) When this function is called from
                    expand_args_fields, we specify any class currently being
                    processed, to make sure we don't try to process a class
                    while it is already being processed.

    Returns:
        new DictConfig object, which is typed.
    """
    if C is None:
        return DictConfig({})

    if _is_configurable_class(C):
        if C in _do_not_process:
            raise ValueError(
                f"Internal recursion error. Need processed {C},"
                f" but cannot get it. _do_not_process={_do_not_process}"
            )
        # This is safe to run multiple times. It will return
        # straight away if C has already been processed.
        expand_args_fields(C, _do_not_process=_do_not_process)

    if dataclasses.is_dataclass(C):
        # Note that if get_default_args_field is used somewhere in C,
        # this call is recursive. No special care is needed,
        # because in practice get_default_args_field is used for
        # separate types than the outer type.

        try:
            out: DictConfig = OmegaConf.structured(C)
        except Exception:
            print(f"### OmegaConf.structured({C}) failed ###")
            # We don't use `raise From` here, because that gets the original
            # exception hidden by the OC_CAUSE logic in the case where we are
            # called by hydra.
            raise
        exclude = getattr(C, "_processed_members", ())
        with open_dict(out):
            for field in exclude:
                out.pop(field, None)
        return out

    if _is_configurable_class(C):
        raise ValueError(f"Failed to process {C}")

    if not inspect.isfunction(C) and not inspect.isclass(C):
        raise ValueError(f"Unexpected {C}")

    dataclass_name = _dataclass_name_for_function(C)
    dataclass = getattr(sys.modules[C.__module__], dataclass_name, None)
    if dataclass is None:
        raise ValueError(
            f"Cannot get args for {C}. Was enable_get_default_args forgotten?"
        )

    try:
        out: DictConfig = OmegaConf.structured(dataclass)
    except Exception:
        print(f"### OmegaConf.structured failed for {C.__name__} ###")
        raise
    return out

def _is_actually_dataclass(some_class) -> bool:
    # Return whether the class some_class has been processed with
    # the dataclass annotation. This is more specific than
    # dataclasses.is_dataclass which returns True on anything
    # deriving from a dataclass.

    # Checking for __init__ would also work for our purpose.
    return "__dataclass_fields__" in some_class.__dict__

def expand_args_fields(
    some_class: Type[_Y], *, _do_not_process: Tuple[type, ...] = ()
) -> Type[_Y]:
    """
    This expands a class which inherits Configurable or ReplaceableBase classes,
    including dataclass processing. some_class is modified in place by this function.
    If expand_args_fields(some_class) has already been called, subsequent calls do
    nothing and return some_class unmodified.
    For classes of type ReplaceableBase, you can add some_class to the registry before
    or after calling this function. But potential inner classes need to be registered
    before this function is run on the outer class.

    The transformations this function makes, before the concluding
    dataclasses.dataclass, are as follows. If X is a base class with registered
    subclasses Y and Z, replace a class member

        x: X

    and optionally

        x_class_type: str = "Y"
        def create_x(self):...

    with

        x_Y_args: dict = dataclasses.field(default_factory=lambda: get_default_args(Y))
        x_Z_args: dict = dataclasses.field(default_factory=lambda: get_default_args(Z))
        def create_x(self):
            args = self.getattr(f"x_{self.x_class_type}_args")
            self.create_x_impl(self.x_class_type, args)
        def create_x_impl(self, x_type, args):
            x_type = registry.get(X, x_type)
            expand_args_fields(x_type)
            self.x = x_type(**args)
        x_class_type: str = "UNDEFAULTED"

    without adding the optional attributes if they are already there.

    Similarly, replace

        x: Optional[X]

    and optionally

        x_class_type: Optional[str] = "Y"
        def create_x(self):...

    with

        x_Y_args: dict = dataclasses.field(default_factory=lambda: get_default_args(Y))
        x_Z_args: dict = dataclasses.field(default_factory=lambda: get_default_args(Z))
        def create_x(self):
            if self.x_class_type is None:
                args = None
            else:
                args = self.getattr(f"x_{self.x_class_type}_args", None)
            self.create_x_impl(self.x_class_type, args)
        def create_x_impl(self, x_class_type, args):
            if x_class_type is None:
                self.x = None
                return

            x_type = registry.get(X, x_class_type)
            expand_args_fields(x_type)
            assert args is not None
            self.x = x_type(**args)
        x_class_type: Optional[str] = "UNDEFAULTED"

    without adding the optional attributes if they are already there.

    Similarly, if X is a subclass of Configurable,

        x: X

    and optionally

        def create_x(self):...

    will be replaced with

        x_args: dict = dataclasses.field(default_factory=lambda: get_default_args(X))
        def create_x(self):
            self.create_x_impl(True, self.x_args)

        def create_x_impl(self, enabled, args):
            if enabled:
                expand_args_fields(X)
                self.x = X(**args)
            else:
                self.x = None

    Similarly, replace,

        x: Optional[X]
        x_enabled: bool = ...

    and optionally

        def create_x(self):...

    with

        x_args: dict = dataclasses.field(default_factory=lambda: get_default_args(X))
        x_enabled: bool = ...
        def create_x(self):
            self.create_x_impl(self.x_enabled, self.x_args)

        def create_x_impl(self, enabled, args):
            if enabled:
                expand_args_fields(X)
                self.x = X(**args)
            else:
                self.x = None


    Also adds the following class members, unannotated so that dataclass
    ignores them.
        - _creation_functions: Tuple[str, ...] of all the create_ functions,
            including those from base classes (not the create_x_impl ones).
        - _known_implementations: Dict[str, Type] containing the classes which
            have been found from the registry.
            (used only to raise a warning if it one has been overwritten)
        - _processed_members: a Dict[str, Any] of all the members which have been
            transformed, with values giving the types they were declared to have.
            (E.g. {"x": X} or {"x": Optional[X]} in the cases above.)

    In addition, if the class has a member function

        @classmethod
        def x_tweak_args(cls, member_type: Type, args: DictConfig) -> None

    then the default_factory of x_args will also have a call to x_tweak_args(X, x_args) and
    the default_factory of x_Y_args will also have a call to x_tweak_args(Y, x_Y_args).

    In addition, if the class inherits torch.nn.Module, the generated __init__ will
    call torch.nn.Module's __init__ before doing anything else.

    Before any transformation of the class, if the class has a classmethod called
    `pre_expand`, it will be called with no arguments.

    Note that although the *_args members are intended to have type DictConfig, they
    are actually internally annotated as dicts. OmegaConf is happy to see a DictConfig
    in place of a dict, but not vice-versa. Allowing dict lets a class user specify
    x_args as an explicit dict without getting an incomprehensible error.

    Args:
        some_class: the class to be processed
        _do_not_process: Internal use for get_default_args: Because get_default_args calls
                        and is called by this function, we let it specify any class currently
                        being processed, to make sure we don't try to process a class while
                        it is already being processed.


    Returns:
        some_class itself, which has been modified in place. This
        allows this function to be used as a class decorator.
    """
    if _is_actually_dataclass(some_class):
        return some_class

    if hasattr(some_class, PRE_EXPAND_NAME):
        getattr(some_class, PRE_EXPAND_NAME)()

    # The functions this class's run_auto_creation will run.
    creation_functions: List[str] = []
    # The classes which this type knows about from the registry
    # We could use a weakref.WeakValueDictionary here which would mean
    # that we don't warn if the class we should have expected is elsewhere
    # unused.
    known_implementations: Dict[str, Type] = {}
    # Names of members which have been processed.
    processed_members: Dict[str, Any] = {}

    # For all bases except ReplaceableBase and Configurable and object,
    # we need to process them before our own processing. This is
    # because dataclasses expect to inherit dataclasses and not unprocessed
    # dataclasses.
    for base in some_class.mro()[-3:0:-1]:
        if base is ReplaceableBase:
            continue
        if base is Configurable:
            continue
        if not issubclass(base, (Configurable, ReplaceableBase)):
            continue
        expand_args_fields(base, _do_not_process=_do_not_process)
        if "_creation_functions" in base.__dict__:
            creation_functions.extend(base._creation_functions)
        if "_known_implementations" in base.__dict__:
            known_implementations.update(base._known_implementations)
        if "_processed_members" in base.__dict__:
            processed_members.update(base._processed_members)

    to_process: List[Tuple[str, Type, _ProcessType]] = []
    if "__annotations__" in some_class.__dict__:
        for name, type_ in some_class.__annotations__.items():
            underlying_and_process_type = _get_type_to_process(type_)
            if underlying_and_process_type is None:
                continue
            underlying_type, process_type = underlying_and_process_type
            to_process.append((name, underlying_type, process_type))

    for name, underlying_type, process_type in to_process:
        processed_members[name] = some_class.__annotations__[name]
        _process_member(
            name=name,
            type_=underlying_type,
            process_type=process_type,
            some_class=some_class,
            creation_functions=creation_functions,
            _do_not_process=_do_not_process,
            known_implementations=known_implementations,
        )

    for key, count in Counter(creation_functions).items():
        if count > 1:
            warnings.warn(f"Clash with {key} in a base class.")
    some_class._creation_functions = tuple(creation_functions)
    some_class._processed_members = processed_members
    some_class._known_implementations = known_implementations

    dataclasses.dataclass(eq=False)(some_class)
    _fixup_class_init(some_class)
    return some_class

