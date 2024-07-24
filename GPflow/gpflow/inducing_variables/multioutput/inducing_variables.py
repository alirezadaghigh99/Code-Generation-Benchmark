class SharedIndependentInducingVariables(FallbackSharedIndependentInducingVariables):
    """
    Here, we define the same inducing variables as in the base class. However,
    this class is intended to be used without the constraints on the shapes that
    `Kuu()` and `Kuf()` return. This allows a custom `conditional()` to provide
    the most efficient implementation.
    """

class SeparateIndependentInducingVariables(FallbackSeparateIndependentInducingVariables):
    """
    Here, we define the same inducing variables as in the base class. However,
    this class is intended to be used without the constraints on the shapes that
    `Kuu()` and `Kuf()` return. This allows a custom `conditional()` to provide
    the most efficient implementation.
    """

