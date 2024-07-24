def default_float() -> type:
    """Returns default float type"""
    return config().float

def default_jitter() -> float:
    """
    The jitter is a constant that GPflow adds to the diagonal of matrices
    to achieve numerical stability of the system when the condition number
    of the associated matrices is large, and therefore the matrices nearly singular.
    """
    return config().jitter

def as_context(temporary_config: Optional[Config] = None) -> Generator[None, None, None]:
    """Ensure that global configs defaults, with a context manager. Useful for testing."""
    current_config = config()
    temporary_config = replace(current_config) if temporary_config is None else temporary_config
    try:
        set_config(temporary_config)
        yield
    finally:
        set_config(current_config)

def set_default_positive_bijector(value: str) -> None:
    """
    Sets positive bijector type.
    There are currently two options implemented: "exp" and "softplus".
    """
    type_map = positive_bijector_type_map()
    if isinstance(value, str):
        value = value.lower()
    if value not in type_map:
        raise ValueError(f"`{value}` not in set of valid bijectors: {sorted(type_map)}")

    set_config(replace(config(), positive_bijector=value))

def set_default_summary_fmt(value: Optional[str]) -> None:
    formats: List[Optional[str]] = list(tabulate.tabulate_formats)
    formats.extend(["notebook", None])
    if value not in formats:
        raise ValueError(f"Summary does not support '{value}' format")

    set_config(replace(config(), summary_fmt=value))

class Config:
    """
    Immutable object for storing global GPflow settings
    """

    int: type = field(default_factory=_default_int_factory)
    """Integer data type, int32 or int64."""

    float: type = field(default_factory=_default_float_factory)
    """Float data type, float32 or float64"""

    jitter: Float = field(default_factory=_default_jitter_factory)
    """
    Jitter value. Mainly used for for making badly conditioned matrices more stable.

    Default value is `1e-6`.
    """

    positive_bijector: str = field(default_factory=_default_positive_bijector_factory)
    """
    Method for positive bijector, either "softplus" or "exp".

    Default is "softplus".
    """

    positive_minimum: Float = field(default_factory=_default_positive_minimum_factory)
    """Lower bound for the positive transformation."""

    likelihood_positive_minimum: Float = field(
        default_factory=_default_likelihood_positive_minimum_factory
    )
    """Lower bound for the positive transformation for positive likelihood parameters."""

    summary_fmt: Optional[str] = field(default_factory=_default_summary_fmt_factory)
    """Summary format for module printing."""

