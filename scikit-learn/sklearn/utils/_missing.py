def is_scalar_nan(x):
    """Test if x is NaN.

    This function is meant to overcome the issue that np.isnan does not allow
    non-numerical types as input, and that np.nan is not float('nan').

    Parameters
    ----------
    x : any type
        Any scalar value.

    Returns
    -------
    bool
        Returns true if x is NaN, and false otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.utils._missing import is_scalar_nan
    >>> is_scalar_nan(np.nan)
    True
    >>> is_scalar_nan(float("nan"))
    True
    >>> is_scalar_nan(None)
    False
    >>> is_scalar_nan("")
    False
    >>> is_scalar_nan([np.nan])
    False
    """
    return (
        not isinstance(x, numbers.Integral)
        and isinstance(x, numbers.Real)
        and math.isnan(x)
    )

