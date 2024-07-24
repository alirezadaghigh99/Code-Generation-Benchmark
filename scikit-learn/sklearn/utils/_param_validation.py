def validate_params(parameter_constraints, *, prefer_skip_nested_validation):
    """Decorator to validate types and values of functions and methods.

    Parameters
    ----------
    parameter_constraints : dict
        A dictionary `param_name: list of constraints`. See the docstring of
        `validate_parameter_constraints` for a description of the accepted constraints.

        Note that the *args and **kwargs parameters are not validated and must not be
        present in the parameter_constraints dictionary.

    prefer_skip_nested_validation : bool
        If True, the validation of parameters of inner estimators or functions
        called by the decorated function will be skipped.

        This is useful to avoid validating many times the parameters passed by the
        user from the public facing API. It's also useful to avoid validating
        parameters that we pass internally to inner functions that are guaranteed to
        be valid by the test suite.

        It should be set to True for most functions, except for those that receive
        non-validated objects as parameters or that are just wrappers around classes
        because they only perform a partial validation.

    Returns
    -------
    decorated_function : function or method
        The decorated function.
    """

    def decorator(func):
        # The dict of parameter constraints is set as an attribute of the function
        # to make it possible to dynamically introspect the constraints for
        # automatic testing.
        setattr(func, "_skl_parameter_constraints", parameter_constraints)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            global_skip_validation = get_config()["skip_parameter_validation"]
            if global_skip_validation:
                return func(*args, **kwargs)

            func_sig = signature(func)

            # Map *args/**kwargs to the function signature
            params = func_sig.bind(*args, **kwargs)
            params.apply_defaults()

            # ignore self/cls and positional/keyword markers
            to_ignore = [
                p.name
                for p in func_sig.parameters.values()
                if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
            ]
            to_ignore += ["self", "cls"]
            params = {k: v for k, v in params.arguments.items() if k not in to_ignore}

            validate_parameter_constraints(
                parameter_constraints, params, caller_name=func.__qualname__
            )

            try:
                with config_context(
                    skip_parameter_validation=(
                        prefer_skip_nested_validation or global_skip_validation
                    )
                ):
                    return func(*args, **kwargs)
            except InvalidParameterError as e:
                # When the function is just a wrapper around an estimator, we allow
                # the function to delegate validation to the estimator, but we replace
                # the name of the estimator by the name of the function in the error
                # message to avoid confusion.
                msg = re.sub(
                    r"parameter of \w+ must be",
                    f"parameter of {func.__qualname__} must be",
                    str(e),
                )
                raise InvalidParameterError(msg) from e

        return wrapper

    return decorator

def make_constraint(constraint):
    """Convert the constraint into the appropriate Constraint object.

    Parameters
    ----------
    constraint : object
        The constraint to convert.

    Returns
    -------
    constraint : instance of _Constraint
        The converted constraint.
    """
    if isinstance(constraint, str) and constraint == "array-like":
        return _ArrayLikes()
    if isinstance(constraint, str) and constraint == "sparse matrix":
        return _SparseMatrices()
    if isinstance(constraint, str) and constraint == "random_state":
        return _RandomStates()
    if constraint is callable:
        return _Callables()
    if constraint is None:
        return _NoneConstraint()
    if isinstance(constraint, type):
        return _InstancesOf(constraint)
    if isinstance(
        constraint, (Interval, StrOptions, Options, HasMethods, MissingValues)
    ):
        return constraint
    if isinstance(constraint, str) and constraint == "boolean":
        return _Booleans()
    if isinstance(constraint, str) and constraint == "verbose":
        return _VerboseHelper()
    if isinstance(constraint, str) and constraint == "cv_object":
        return _CVObjects()
    if isinstance(constraint, Hidden):
        constraint = make_constraint(constraint.constraint)
        constraint.hidden = True
        return constraint
    if isinstance(constraint, str) and constraint == "nan":
        return _NanConstraint()
    raise ValueError(f"Unknown constraint type: {constraint}")

class Interval(_Constraint):
    """Constraint representing a typed interval.

    Parameters
    ----------
    type : {numbers.Integral, numbers.Real, RealNotInt}
        The set of numbers in which to set the interval.

        If RealNotInt, only reals that don't have the integer type
        are allowed. For example 1.0 is allowed but 1 is not.

    left : float or int or None
        The left bound of the interval. None means left bound is -∞.

    right : float, int or None
        The right bound of the interval. None means right bound is +∞.

    closed : {"left", "right", "both", "neither"}
        Whether the interval is open or closed. Possible choices are:

        - `"left"`: the interval is closed on the left and open on the right.
          It is equivalent to the interval `[ left, right )`.
        - `"right"`: the interval is closed on the right and open on the left.
          It is equivalent to the interval `( left, right ]`.
        - `"both"`: the interval is closed.
          It is equivalent to the interval `[ left, right ]`.
        - `"neither"`: the interval is open.
          It is equivalent to the interval `( left, right )`.

    Notes
    -----
    Setting a bound to `None` and setting the interval closed is valid. For instance,
    strictly speaking, `Interval(Real, 0, None, closed="both")` corresponds to
    `[0, +∞) U {+∞}`.
    """

    def __init__(self, type, left, right, *, closed):
        super().__init__()
        self.type = type
        self.left = left
        self.right = right
        self.closed = closed

        self._check_params()

    def _check_params(self):
        if self.type not in (Integral, Real, RealNotInt):
            raise ValueError(
                "type must be either numbers.Integral, numbers.Real or RealNotInt."
                f" Got {self.type} instead."
            )

        if self.closed not in ("left", "right", "both", "neither"):
            raise ValueError(
                "closed must be either 'left', 'right', 'both' or 'neither'. "
                f"Got {self.closed} instead."
            )

        if self.type is Integral:
            suffix = "for an interval over the integers."
            if self.left is not None and not isinstance(self.left, Integral):
                raise TypeError(f"Expecting left to be an int {suffix}")
            if self.right is not None and not isinstance(self.right, Integral):
                raise TypeError(f"Expecting right to be an int {suffix}")
            if self.left is None and self.closed in ("left", "both"):
                raise ValueError(
                    f"left can't be None when closed == {self.closed} {suffix}"
                )
            if self.right is None and self.closed in ("right", "both"):
                raise ValueError(
                    f"right can't be None when closed == {self.closed} {suffix}"
                )
        else:
            if self.left is not None and not isinstance(self.left, Real):
                raise TypeError("Expecting left to be a real number.")
            if self.right is not None and not isinstance(self.right, Real):
                raise TypeError("Expecting right to be a real number.")

        if self.right is not None and self.left is not None and self.right <= self.left:
            raise ValueError(
                f"right can't be less than left. Got left={self.left} and "
                f"right={self.right}"
            )

    def __contains__(self, val):
        if not isinstance(val, Integral) and np.isnan(val):
            return False

        left_cmp = operator.lt if self.closed in ("left", "both") else operator.le
        right_cmp = operator.gt if self.closed in ("right", "both") else operator.ge

        left = -np.inf if self.left is None else self.left
        right = np.inf if self.right is None else self.right

        if left_cmp(val, left):
            return False
        if right_cmp(val, right):
            return False
        return True

    def is_satisfied_by(self, val):
        if not isinstance(val, self.type):
            return False

        return val in self

    def __str__(self):
        type_str = "an int" if self.type is Integral else "a float"
        left_bracket = "[" if self.closed in ("left", "both") else "("
        left_bound = "-inf" if self.left is None else self.left
        right_bound = "inf" if self.right is None else self.right
        right_bracket = "]" if self.closed in ("right", "both") else ")"

        # better repr if the bounds were given as integers
        if not self.type == Integral and isinstance(self.left, Real):
            left_bound = float(left_bound)
        if not self.type == Integral and isinstance(self.right, Real):
            right_bound = float(right_bound)

        return (
            f"{type_str} in the range "
            f"{left_bracket}{left_bound}, {right_bound}{right_bracket}"
        )

class StrOptions(Options):
    """Constraint representing a finite set of strings.

    Parameters
    ----------
    options : set of str
        The set of valid strings.

    deprecated : set of str or None, default=None
        A subset of the `options` to mark as deprecated in the string
        representation of the constraint.
    """

    def __init__(self, options, *, deprecated=None):
        super().__init__(type=str, options=options, deprecated=deprecated)

class HasMethods(_Constraint):
    """Constraint representing objects that expose specific methods.

    It is useful for parameters following a protocol and where we don't want to impose
    an affiliation to a specific module or class.

    Parameters
    ----------
    methods : str or list of str
        The method(s) that the object is expected to expose.
    """

    @validate_params(
        {"methods": [str, list]},
        prefer_skip_nested_validation=True,
    )
    def __init__(self, methods):
        super().__init__()
        if isinstance(methods, str):
            methods = [methods]
        self.methods = methods

    def is_satisfied_by(self, val):
        return all(callable(getattr(val, method, None)) for method in self.methods)

    def __str__(self):
        if len(self.methods) == 1:
            methods = f"{self.methods[0]!r}"
        else:
            methods = (
                f"{', '.join([repr(m) for m in self.methods[:-1]])} and"
                f" {self.methods[-1]!r}"
            )
        return f"an object implementing {methods}"

class _InstancesOf(_Constraint):
    """Constraint representing instances of a given type.

    Parameters
    ----------
    type : type
        The valid type.
    """

    def __init__(self, type):
        super().__init__()
        self.type = type

    def is_satisfied_by(self, val):
        return isinstance(val, self.type)

    def __str__(self):
        return f"an instance of {_type_name(self.type)!r}"

class Options(_Constraint):
    """Constraint representing a finite set of instances of a given type.

    Parameters
    ----------
    type : type

    options : set
        The set of valid scalars.

    deprecated : set or None, default=None
        A subset of the `options` to mark as deprecated in the string
        representation of the constraint.
    """

    def __init__(self, type, options, *, deprecated=None):
        super().__init__()
        self.type = type
        self.options = options
        self.deprecated = deprecated or set()

        if self.deprecated - self.options:
            raise ValueError("The deprecated options must be a subset of the options.")

    def is_satisfied_by(self, val):
        return isinstance(val, self.type) and val in self.options

    def _mark_if_deprecated(self, option):
        """Add a deprecated mark to an option if needed."""
        option_str = f"{option!r}"
        if option in self.deprecated:
            option_str = f"{option_str} (deprecated)"
        return option_str

    def __str__(self):
        options_str = (
            f"{', '.join([self._mark_if_deprecated(o) for o in self.options])}"
        )
        return f"a {_type_name(self.type)} among {{{options_str}}}"

class _IterablesNotString(_Constraint):
    """Constraint representing iterables that are not strings."""

    def is_satisfied_by(self, val):
        return isinstance(val, Iterable) and not isinstance(val, str)

    def __str__(self):
        return "an iterable"

class _PandasNAConstraint(_Constraint):
    """Constraint representing the indicator `pd.NA`."""

    def is_satisfied_by(self, val):
        try:
            import pandas as pd

            return isinstance(val, type(pd.NA)) and pd.isna(val)
        except ImportError:
            return False

    def __str__(self):
        return "pandas.NA"

