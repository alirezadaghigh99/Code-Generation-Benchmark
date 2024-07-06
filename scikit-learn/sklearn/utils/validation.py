def check_array(
    array,
    accept_sparse=False,
    *,
    accept_large_sparse=True,
    dtype="numeric",
    order=None,
    copy=False,
    force_writeable=False,
    force_all_finite=True,
    ensure_2d=True,
    allow_nd=False,
    ensure_min_samples=1,
    ensure_min_features=1,
    estimator=None,
    input_name="",
):
    """Input validation on an array, list, sparse matrix or similar.

    By default, the input is checked to be a non-empty 2D array containing
    only finite values. If the dtype of the array is object, attempt
    converting to float, raising on failure.

    Parameters
    ----------
    array : object
        Input object to check / convert.

    accept_sparse : str, bool or list/tuple of str, default=False
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.

    accept_large_sparse : bool, default=True
        If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
        accept_sparse, accept_large_sparse=False will cause it to be accepted
        only if its indices are stored with a 32-bit dtype.

        .. versionadded:: 0.20

    dtype : 'numeric', type, list of type or None, default='numeric'
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.

    order : {'F', 'C'} or None, default=None
        Whether an array will be forced to be fortran or c-style.
        When order is None (default), then if copy=False, nothing is ensured
        about the memory layout of the output array; otherwise (copy=True)
        the memory layout of the returned array is kept as close as possible
        to the original array.

    copy : bool, default=False
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_writeable : bool, default=False
        Whether to force the output array to be writeable. If True, the returned array
        is guaranteed to be writeable, which may require a copy. Otherwise the
        writeability of the input array is preserved.

        .. versionadded:: 1.6

    force_all_finite : bool or 'allow-nan', default=True
        Whether to raise an error on np.inf, np.nan, pd.NA in array. The
        possibilities are:

        - True: Force all values of array to be finite.
        - False: accepts np.inf, np.nan, pd.NA in array.
        - 'allow-nan': accepts only np.nan and pd.NA values in array. Values
          cannot be infinite.

        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.

        .. versionchanged:: 0.23
           Accepts `pd.NA` and converts it into `np.nan`

    ensure_2d : bool, default=True
        Whether to raise a value error if array is not 2D.

    allow_nd : bool, default=False
        Whether to allow array.ndim > 2.

    ensure_min_samples : int, default=1
        Make sure that the array has a minimum number of samples in its first
        axis (rows for a 2D array). Setting to 0 disables this check.

    ensure_min_features : int, default=1
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when the input data has effectively 2
        dimensions or is originally 1D and ``ensure_2d`` is True. Setting to 0
        disables this check.

    estimator : str or estimator instance, default=None
        If passed, include the name of the estimator in warning messages.

    input_name : str, default=""
        The data name used to construct the error message. In particular
        if `input_name` is "X" and the data has NaN values and
        allow_nan is False, the error message will link to the imputer
        documentation.

        .. versionadded:: 1.1.0

    Returns
    -------
    array_converted : object
        The converted and validated array.

    Examples
    --------
    >>> from sklearn.utils.validation import check_array
    >>> X = [[1, 2, 3], [4, 5, 6]]
    >>> X_checked = check_array(X)
    >>> X_checked
    array([[1, 2, 3], [4, 5, 6]])
    """
    if isinstance(array, np.matrix):
        raise TypeError(
            "np.matrix is not supported. Please convert to a numpy array with "
            "np.asarray. For more information see: "
            "https://numpy.org/doc/stable/reference/generated/numpy.matrix.html"
        )

    xp, is_array_api_compliant = get_namespace(array)

    # store reference to original array to check if copy is needed when
    # function returns
    array_orig = array

    # store whether originally we wanted numeric dtype
    dtype_numeric = isinstance(dtype, str) and dtype == "numeric"

    dtype_orig = getattr(array, "dtype", None)
    if not is_array_api_compliant and not hasattr(dtype_orig, "kind"):
        # not a data type (e.g. a column named dtype in a pandas DataFrame)
        dtype_orig = None

    # check if the object contains several dtypes (typically a pandas
    # DataFrame), and store them. If not, store None.
    dtypes_orig = None
    pandas_requires_conversion = False
    # track if we have a Series-like object to raise a better error message
    type_if_series = None
    if hasattr(array, "dtypes") and hasattr(array.dtypes, "__array__"):
        # throw warning if columns are sparse. If all columns are sparse, then
        # array.sparse exists and sparsity will be preserved (later).
        with suppress(ImportError):
            from pandas import SparseDtype

            def is_sparse(dtype):
                return isinstance(dtype, SparseDtype)

            if not hasattr(array, "sparse") and array.dtypes.apply(is_sparse).any():
                warnings.warn(
                    "pandas.DataFrame with sparse columns found."
                    "It will be converted to a dense numpy array."
                )

        dtypes_orig = list(array.dtypes)
        pandas_requires_conversion = any(
            _pandas_dtype_needs_early_conversion(i) for i in dtypes_orig
        )
        if all(isinstance(dtype_iter, np.dtype) for dtype_iter in dtypes_orig):
            dtype_orig = np.result_type(*dtypes_orig)
        elif pandas_requires_conversion and any(d == object for d in dtypes_orig):
            # Force object if any of the dtypes is an object
            dtype_orig = object

    elif (_is_extension_array_dtype(array) or hasattr(array, "iloc")) and hasattr(
        array, "dtype"
    ):
        # array is a pandas series
        type_if_series = type(array)
        pandas_requires_conversion = _pandas_dtype_needs_early_conversion(array.dtype)
        if isinstance(array.dtype, np.dtype):
            dtype_orig = array.dtype
        else:
            # Set to None to let array.astype work out the best dtype
            dtype_orig = None

    if dtype_numeric:
        if (
            dtype_orig is not None
            and hasattr(dtype_orig, "kind")
            and dtype_orig.kind == "O"
        ):
            # if input is object, convert to float.
            dtype = xp.float64
        else:
            dtype = None

    if isinstance(dtype, (list, tuple)):
        if dtype_orig is not None and dtype_orig in dtype:
            # no dtype conversion required
            dtype = None
        else:
            # dtype conversion required. Let's select the first element of the
            # list of accepted types.
            dtype = dtype[0]

    if pandas_requires_conversion:
        # pandas dataframe requires conversion earlier to handle extension dtypes with
        # nans
        # Use the original dtype for conversion if dtype is None
        new_dtype = dtype_orig if dtype is None else dtype
        array = array.astype(new_dtype)
        # Since we converted here, we do not need to convert again later
        dtype = None

    if force_all_finite not in (True, False, "allow-nan"):
        raise ValueError(
            'force_all_finite should be a bool or "allow-nan". Got {!r} instead'.format(
                force_all_finite
            )
        )

    if dtype is not None and _is_numpy_namespace(xp):
        # convert to dtype object to conform to Array API to be use `xp.isdtype` later
        dtype = np.dtype(dtype)

    estimator_name = _check_estimator_name(estimator)
    context = " by %s" % estimator_name if estimator is not None else ""

    # When all dataframe columns are sparse, convert to a sparse array
    if hasattr(array, "sparse") and array.ndim > 1:
        with suppress(ImportError):
            from pandas import SparseDtype  # noqa: F811

            def is_sparse(dtype):
                return isinstance(dtype, SparseDtype)

            if array.dtypes.apply(is_sparse).all():
                # DataFrame.sparse only supports `to_coo`
                array = array.sparse.to_coo()
                if array.dtype == np.dtype("object"):
                    unique_dtypes = set([dt.subtype.name for dt in array_orig.dtypes])
                    if len(unique_dtypes) > 1:
                        raise ValueError(
                            "Pandas DataFrame with mixed sparse extension arrays "
                            "generated a sparse matrix with object dtype which "
                            "can not be converted to a scipy sparse matrix."
                            "Sparse extension arrays should all have the same "
                            "numeric type."
                        )

    if sp.issparse(array):
        _ensure_no_complex_data(array)
        array = _ensure_sparse_format(
            array,
            accept_sparse=accept_sparse,
            dtype=dtype,
            copy=copy,
            force_all_finite=force_all_finite,
            accept_large_sparse=accept_large_sparse,
            estimator_name=estimator_name,
            input_name=input_name,
        )
        if ensure_2d and array.ndim < 2:
            raise ValueError(
                f"Expected 2D input, got input with shape {array.shape}.\n"
                "Reshape your data either using array.reshape(-1, 1) if "
                "your data has a single feature or array.reshape(1, -1) "
                "if it contains a single sample."
            )
    else:
        # If np.array(..) gives ComplexWarning, then we convert the warning
        # to an error. This is needed because specifying a non complex
        # dtype to the function converts complex to real dtype,
        # thereby passing the test made in the lines following the scope
        # of warnings context manager.
        with warnings.catch_warnings():
            try:
                warnings.simplefilter("error", ComplexWarning)
                if dtype is not None and xp.isdtype(dtype, "integral"):
                    # Conversion float -> int should not contain NaN or
                    # inf (numpy#14412). We cannot use casting='safe' because
                    # then conversion float -> int would be disallowed.
                    array = _asarray_with_order(array, order=order, xp=xp)
                    if xp.isdtype(array.dtype, ("real floating", "complex floating")):
                        _assert_all_finite(
                            array,
                            allow_nan=False,
                            msg_dtype=dtype,
                            estimator_name=estimator_name,
                            input_name=input_name,
                        )
                    array = xp.astype(array, dtype, copy=False)
                else:
                    array = _asarray_with_order(array, order=order, dtype=dtype, xp=xp)
            except ComplexWarning as complex_warning:
                raise ValueError(
                    "Complex data not supported\n{}\n".format(array)
                ) from complex_warning

        # It is possible that the np.array(..) gave no warning. This happens
        # when no dtype conversion happened, for example dtype = None. The
        # result is that np.array(..) produces an array of complex dtype
        # and we need to catch and raise exception for such cases.
        _ensure_no_complex_data(array)

        if ensure_2d:
            # If input is scalar raise error
            if array.ndim == 0:
                raise ValueError(
                    "Expected 2D array, got scalar array instead:\narray={}.\n"
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample.".format(array)
                )
            # If input is 1D raise error
            if array.ndim == 1:
                # If input is a Series-like object (eg. pandas Series or polars Series)
                if type_if_series is not None:
                    msg = (
                        f"Expected a 2-dimensional container but got {type_if_series} "
                        "instead. Pass a DataFrame containing a single row (i.e. "
                        "single sample) or a single column (i.e. single feature) "
                        "instead."
                    )
                else:
                    msg = (
                        f"Expected 2D array, got 1D array instead:\narray={array}.\n"
                        "Reshape your data either using array.reshape(-1, 1) if "
                        "your data has a single feature or array.reshape(1, -1) "
                        "if it contains a single sample."
                    )
                raise ValueError(msg)

        if dtype_numeric and hasattr(array.dtype, "kind") and array.dtype.kind in "USV":
            raise ValueError(
                "dtype='numeric' is not compatible with arrays of bytes/strings."
                "Convert your data to numeric values explicitly instead."
            )
        if not allow_nd and array.ndim >= 3:
            raise ValueError(
                "Found array with dim %d. %s expected <= 2."
                % (array.ndim, estimator_name)
            )

        if force_all_finite:
            _assert_all_finite(
                array,
                input_name=input_name,
                estimator_name=estimator_name,
                allow_nan=force_all_finite == "allow-nan",
            )

        if copy:
            if _is_numpy_namespace(xp):
                # only make a copy if `array` and `array_orig` may share memory`
                if np.may_share_memory(array, array_orig):
                    array = _asarray_with_order(
                        array, dtype=dtype, order=order, copy=True, xp=xp
                    )
            else:
                # always make a copy for non-numpy arrays
                array = _asarray_with_order(
                    array, dtype=dtype, order=order, copy=True, xp=xp
                )

    if ensure_min_samples > 0:
        n_samples = _num_samples(array)
        if n_samples < ensure_min_samples:
            raise ValueError(
                "Found array with %d sample(s) (shape=%s) while a"
                " minimum of %d is required%s."
                % (n_samples, array.shape, ensure_min_samples, context)
            )

    if ensure_min_features > 0 and array.ndim == 2:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError(
                "Found array with %d feature(s) (shape=%s) while"
                " a minimum of %d is required%s."
                % (n_features, array.shape, ensure_min_features, context)
            )

    if force_writeable:
        # By default, array.copy() creates a C-ordered copy. We set order=K to
        # preserve the order of the array.
        copy_params = {"order": "K"} if not sp.issparse(array) else {}

        array_data = array.data if sp.issparse(array) else array
        flags = getattr(array_data, "flags", None)
        if not getattr(flags, "writeable", True):
            # This situation can only happen when copy=False, the array is read-only and
            # a writeable output is requested. This is an ambiguous setting so we chose
            # to always (except for one specific setting, see below) make a copy to
            # ensure that the output is writeable, even if avoidable, to not overwrite
            # the user's data by surprise.

            if _is_pandas_df_or_series(array_orig):
                try:
                    # In pandas >= 3, np.asarray(df), called earlier in check_array,
                    # returns a read-only intermediate array. It can be made writeable
                    # safely without copy because if the original DataFrame was backed
                    # by a read-only array, trying to change the flag would raise an
                    # error, in which case we make a copy.
                    array_data.flags.writeable = True
                except ValueError:
                    array = array.copy(**copy_params)
            else:
                array = array.copy(**copy_params)

    return array

def _num_samples(x):
    """Return number of samples in array-like x."""
    message = "Expected sequence or array-like, got %s" % type(x)
    if hasattr(x, "fit") and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError(message)

    if _use_interchange_protocol(x):
        return x.__dataframe__().num_rows()

    if not hasattr(x, "__len__") and not hasattr(x, "shape"):
        if hasattr(x, "__array__"):
            x = np.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, "shape") and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError(
                "Singleton array %r cannot be considered a valid collection." % x
            )
        # Check that shape is returning an integer or default to len
        # Dask dataframes may not return numeric shape[0] value
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]

    try:
        return len(x)
    except TypeError as type_error:
        raise TypeError(message) from type_error

def _get_feature_names(X):
    """Get feature names from X.

    Support for other array containers should place its implementation here.

    Parameters
    ----------
    X : {ndarray, dataframe} of shape (n_samples, n_features)
        Array container to extract feature names.

        - pandas dataframe : The columns will be considered to be feature
          names. If the dataframe contains non-string feature names, `None` is
          returned.
        - All other array containers will return `None`.

    Returns
    -------
    names: ndarray or None
        Feature names of `X`. Unrecognized array containers will return `None`.
    """
    feature_names = None

    # extract feature names for support array containers
    if _is_pandas_df(X):
        # Make sure we can inspect columns names from pandas, even with
        # versions too old to expose a working implementation of
        # __dataframe__.column_names() and avoid introducing any
        # additional copy.
        # TODO: remove the pandas-specific branch once the minimum supported
        # version of pandas has a working implementation of
        # __dataframe__.column_names() that is guaranteed to not introduce any
        # additional copy of the data without having to impose allow_copy=False
        # that could fail with other libraries. Note: in the longer term, we
        # could decide to instead rely on the __dataframe_namespace__ API once
        # adopted by our minimally supported pandas version.
        feature_names = np.asarray(X.columns, dtype=object)
    elif hasattr(X, "__dataframe__"):
        df_protocol = X.__dataframe__()
        feature_names = np.asarray(list(df_protocol.column_names()), dtype=object)

    if feature_names is None or len(feature_names) == 0:
        return

    types = sorted(t.__qualname__ for t in set(type(v) for v in feature_names))

    # mixed type of string and non-string is not supported
    if len(types) > 1 and "str" in types:
        raise TypeError(
            "Feature names are only supported if all input features have string names, "
            f"but your input has {types} as feature name / column name types. "
            "If you want feature names to be stored and validated, you must convert "
            "them all to strings, by using X.columns = X.columns.astype(str) for "
            "example. Otherwise you can remove feature / column names from your input "
            "data, or convert them all to a non-string data type."
        )

    # Only feature names of all strings are supported
    if len(types) == 1 and types[0] == "str":
        return feature_names

def _allclose_dense_sparse(x, y, rtol=1e-7, atol=1e-9):
    """Check allclose for sparse and dense data.

    Both x and y need to be either sparse or dense, they
    can't be mixed.

    Parameters
    ----------
    x : {array-like, sparse matrix}
        First array to compare.

    y : {array-like, sparse matrix}
        Second array to compare.

    rtol : float, default=1e-7
        Relative tolerance; see numpy.allclose.

    atol : float, default=1e-9
        absolute tolerance; see numpy.allclose. Note that the default here is
        more tolerant than the default for numpy.testing.assert_allclose, where
        atol=0.
    """
    if sp.issparse(x) and sp.issparse(y):
        x = x.tocsr()
        y = y.tocsr()
        x.sum_duplicates()
        y.sum_duplicates()
        return (
            np.array_equal(x.indices, y.indices)
            and np.array_equal(x.indptr, y.indptr)
            and np.allclose(x.data, y.data, rtol=rtol, atol=atol)
        )
    elif not sp.issparse(x) and not sp.issparse(y):
        return np.allclose(x, y, rtol=rtol, atol=atol)
    raise ValueError(
        "Can only compare two sparse matrices, not a sparse matrix and an array"
    )

def _is_pandas_df(X):
    """Return True if the X is a pandas dataframe."""
    try:
        pd = sys.modules["pandas"]
    except KeyError:
        return False
    return isinstance(X, pd.DataFrame)

def check_scalar(
    x,
    name,
    target_type,
    *,
    min_val=None,
    max_val=None,
    include_boundaries="both",
):
    """Validate scalar parameters type and value.

    Parameters
    ----------
    x : object
        The scalar parameter to validate.

    name : str
        The name of the parameter to be printed in error messages.

    target_type : type or tuple
        Acceptable data types for the parameter.

    min_val : float or int, default=None
        The minimum valid value the parameter can take. If None (default) it
        is implied that the parameter does not have a lower bound.

    max_val : float or int, default=None
        The maximum valid value the parameter can take. If None (default) it
        is implied that the parameter does not have an upper bound.

    include_boundaries : {"left", "right", "both", "neither"}, default="both"
        Whether the interval defined by `min_val` and `max_val` should include
        the boundaries. Possible choices are:

        - `"left"`: only `min_val` is included in the valid interval.
          It is equivalent to the interval `[ min_val, max_val )`.
        - `"right"`: only `max_val` is included in the valid interval.
          It is equivalent to the interval `( min_val, max_val ]`.
        - `"both"`: `min_val` and `max_val` are included in the valid interval.
          It is equivalent to the interval `[ min_val, max_val ]`.
        - `"neither"`: neither `min_val` nor `max_val` are included in the
          valid interval. It is equivalent to the interval `( min_val, max_val )`.

    Returns
    -------
    x : numbers.Number
        The validated number.

    Raises
    ------
    TypeError
        If the parameter's type does not match the desired type.

    ValueError
        If the parameter's value violates the given bounds.
        If `min_val`, `max_val` and `include_boundaries` are inconsistent.

    Examples
    --------
    >>> from sklearn.utils.validation import check_scalar
    >>> check_scalar(10, "x", int, min_val=1, max_val=20)
    10
    """

    def type_name(t):
        """Convert type into humman readable string."""
        module = t.__module__
        qualname = t.__qualname__
        if module == "builtins":
            return qualname
        elif t == numbers.Real:
            return "float"
        elif t == numbers.Integral:
            return "int"
        return f"{module}.{qualname}"

    if not isinstance(x, target_type):
        if isinstance(target_type, tuple):
            types_str = ", ".join(type_name(t) for t in target_type)
            target_type_str = f"{{{types_str}}}"
        else:
            target_type_str = type_name(target_type)

        raise TypeError(
            f"{name} must be an instance of {target_type_str}, not"
            f" {type(x).__qualname__}."
        )

    expected_include_boundaries = ("left", "right", "both", "neither")
    if include_boundaries not in expected_include_boundaries:
        raise ValueError(
            f"Unknown value for `include_boundaries`: {repr(include_boundaries)}. "
            f"Possible values are: {expected_include_boundaries}."
        )

    if max_val is None and include_boundaries == "right":
        raise ValueError(
            "`include_boundaries`='right' without specifying explicitly `max_val` "
            "is inconsistent."
        )

    if min_val is None and include_boundaries == "left":
        raise ValueError(
            "`include_boundaries`='left' without specifying explicitly `min_val` "
            "is inconsistent."
        )

    comparison_operator = (
        operator.lt if include_boundaries in ("left", "both") else operator.le
    )
    if min_val is not None and comparison_operator(x, min_val):
        raise ValueError(
            f"{name} == {x}, must be"
            f" {'>=' if include_boundaries in ('left', 'both') else '>'} {min_val}."
        )

    comparison_operator = (
        operator.gt if include_boundaries in ("right", "both") else operator.ge
    )
    if max_val is not None and comparison_operator(x, max_val):
        raise ValueError(
            f"{name} == {x}, must be"
            f" {'<=' if include_boundaries in ('right', 'both') else '<'} {max_val}."
        )

    return x

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    :class:`numpy:numpy.random.RandomState`
        The random state object based on `seed` parameter.

    Examples
    --------
    >>> from sklearn.utils.validation import check_random_state
    >>> check_random_state(42)
    RandomState(MT19937) at 0x...
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )

def as_float_array(X, *, copy=True, force_all_finite=True):
    """Convert an array-like to an array of floats.

    The new dtype will be np.float32 or np.float64, depending on the original
    type. The function can create a copy or modify the argument depending
    on the argument copy.

    Parameters
    ----------
    X : {array-like, sparse matrix}
        The input data.

    copy : bool, default=True
        If True, a copy of X will be created. If False, a copy may still be
        returned if X's dtype is not a floating point type.

    force_all_finite : bool or 'allow-nan', default=True
        Whether to raise an error on np.inf, np.nan, pd.NA in X. The
        possibilities are:

        - True: Force all values of X to be finite.
        - False: accepts np.inf, np.nan, pd.NA in X.
        - 'allow-nan': accepts only np.nan and pd.NA values in X. Values cannot
          be infinite.

        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.

        .. versionchanged:: 0.23
           Accepts `pd.NA` and converts it into `np.nan`

    Returns
    -------
    XT : {ndarray, sparse matrix}
        An array of type float.

    Examples
    --------
    >>> from sklearn.utils import as_float_array
    >>> import numpy as np
    >>> array = np.array([0, 0, 1, 2, 2], dtype=np.int64)
    >>> as_float_array(array)
    array([0., 0., 1., 2., 2.])
    """
    if isinstance(X, np.matrix) or (
        not isinstance(X, np.ndarray) and not sp.issparse(X)
    ):
        return check_array(
            X,
            accept_sparse=["csr", "csc", "coo"],
            dtype=np.float64,
            copy=copy,
            force_all_finite=force_all_finite,
            ensure_2d=False,
        )
    elif sp.issparse(X) and X.dtype in [np.float32, np.float64]:
        return X.copy() if copy else X
    elif X.dtype in [np.float32, np.float64]:  # is numpy array
        return X.copy("F" if X.flags["F_CONTIGUOUS"] else "C") if copy else X
    else:
        if X.dtype.kind in "uib" and X.dtype.itemsize <= 4:
            return_dtype = np.float32
        else:
            return_dtype = np.float64
        return X.astype(return_dtype)

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    :class:`numpy:numpy.random.RandomState`
        The random state object based on `seed` parameter.

    Examples
    --------
    >>> from sklearn.utils.validation import check_random_state
    >>> check_random_state(42)
    RandomState(MT19937) at 0x...
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )

def _num_features(X):
    """Return the number of features in an array-like X.

    This helper function tries hard to avoid to materialize an array version
    of X unless necessary. For instance, if X is a list of lists,
    this function will return the length of the first element, assuming
    that subsequent elements are all lists of the same length without
    checking.
    Parameters
    ----------
    X : array-like
        array-like to get the number of features.

    Returns
    -------
    features : int
        Number of features
    """
    type_ = type(X)
    if type_.__module__ == "builtins":
        type_name = type_.__qualname__
    else:
        type_name = f"{type_.__module__}.{type_.__qualname__}"
    message = f"Unable to find the number of features from X of type {type_name}"
    if not hasattr(X, "__len__") and not hasattr(X, "shape"):
        if not hasattr(X, "__array__"):
            raise TypeError(message)
        # Only convert X to a numpy array if there is no cheaper, heuristic
        # option.
        X = np.asarray(X)

    if hasattr(X, "shape"):
        if not hasattr(X.shape, "__len__") or len(X.shape) <= 1:
            message += f" with shape {X.shape}"
            raise TypeError(message)
        return X.shape[1]

    first_sample = X[0]

    # Do not consider an array-like of strings or dicts to be a 2D array
    if isinstance(first_sample, (str, bytes, dict)):
        message += f" where the samples are of type {type(first_sample).__qualname__}"
        raise TypeError(message)

    try:
        # If X is a list of lists, for instance, we assume that all nested
        # lists have the same length without checking or converting to
        # a numpy array to keep this function call as cheap as possible.
        return len(first_sample)
    except Exception as err:
        raise TypeError(message) from err

def _check_psd_eigenvalues(lambdas, enable_warnings=False):
    """Check the eigenvalues of a positive semidefinite (PSD) matrix.

    Checks the provided array of PSD matrix eigenvalues for numerical or
    conditioning issues and returns a fixed validated version. This method
    should typically be used if the PSD matrix is user-provided (e.g. a
    Gram matrix) or computed using a user-provided dissimilarity metric
    (e.g. kernel function), or if the decomposition process uses approximation
    methods (randomized SVD, etc.).

    It checks for three things:

    - that there are no significant imaginary parts in eigenvalues (more than
      1e-5 times the maximum real part). If this check fails, it raises a
      ``ValueError``. Otherwise all non-significant imaginary parts that may
      remain are set to zero. This operation is traced with a
      ``PositiveSpectrumWarning`` when ``enable_warnings=True``.

    - that eigenvalues are not all negative. If this check fails, it raises a
      ``ValueError``

    - that there are no significant negative eigenvalues with absolute value
      more than 1e-10 (1e-6) and more than 1e-5 (5e-3) times the largest
      positive eigenvalue in double (simple) precision. If this check fails,
      it raises a ``ValueError``. Otherwise all negative eigenvalues that may
      remain are set to zero. This operation is traced with a
      ``PositiveSpectrumWarning`` when ``enable_warnings=True``.

    Finally, all the positive eigenvalues that are too small (with a value
    smaller than the maximum eigenvalue multiplied by 1e-12 (2e-7)) are set to
    zero. This operation is traced with a ``PositiveSpectrumWarning`` when
    ``enable_warnings=True``.

    Parameters
    ----------
    lambdas : array-like of shape (n_eigenvalues,)
        Array of eigenvalues to check / fix.

    enable_warnings : bool, default=False
        When this is set to ``True``, a ``PositiveSpectrumWarning`` will be
        raised when there are imaginary parts, negative eigenvalues, or
        extremely small non-zero eigenvalues. Otherwise no warning will be
        raised. In both cases, imaginary parts, negative eigenvalues, and
        extremely small non-zero eigenvalues will be set to zero.

    Returns
    -------
    lambdas_fixed : ndarray of shape (n_eigenvalues,)
        A fixed validated copy of the array of eigenvalues.

    Examples
    --------
    >>> from sklearn.utils.validation import _check_psd_eigenvalues
    >>> _check_psd_eigenvalues([1, 2])      # nominal case
    array([1, 2])
    >>> _check_psd_eigenvalues([5, 5j])     # significant imag part
    Traceback (most recent call last):
        ...
    ValueError: There are significant imaginary parts in eigenvalues (1
        of the maximum real part). Either the matrix is not PSD, or there was
        an issue while computing the eigendecomposition of the matrix.
    >>> _check_psd_eigenvalues([5, 5e-5j])  # insignificant imag part
    array([5., 0.])
    >>> _check_psd_eigenvalues([-5, -1])    # all negative
    Traceback (most recent call last):
        ...
    ValueError: All eigenvalues are negative (maximum is -1). Either the
        matrix is not PSD, or there was an issue while computing the
        eigendecomposition of the matrix.
    >>> _check_psd_eigenvalues([5, -1])     # significant negative
    Traceback (most recent call last):
        ...
    ValueError: There are significant negative eigenvalues (0.2 of the
        maximum positive). Either the matrix is not PSD, or there was an issue
        while computing the eigendecomposition of the matrix.
    >>> _check_psd_eigenvalues([5, -5e-5])  # insignificant negative
    array([5., 0.])
    >>> _check_psd_eigenvalues([5, 4e-12])  # bad conditioning (too small)
    array([5., 0.])

    """

    lambdas = np.array(lambdas)
    is_double_precision = lambdas.dtype == np.float64

    # note: the minimum value available is
    #  - single-precision: np.finfo('float32').eps = 1.2e-07
    #  - double-precision: np.finfo('float64').eps = 2.2e-16

    # the various thresholds used for validation
    # we may wish to change the value according to precision.
    significant_imag_ratio = 1e-5
    significant_neg_ratio = 1e-5 if is_double_precision else 5e-3
    significant_neg_value = 1e-10 if is_double_precision else 1e-6
    small_pos_ratio = 1e-12 if is_double_precision else 2e-7

    # Check that there are no significant imaginary parts
    if not np.isreal(lambdas).all():
        max_imag_abs = np.abs(np.imag(lambdas)).max()
        max_real_abs = np.abs(np.real(lambdas)).max()
        if max_imag_abs > significant_imag_ratio * max_real_abs:
            raise ValueError(
                "There are significant imaginary parts in eigenvalues (%g "
                "of the maximum real part). Either the matrix is not PSD, or "
                "there was an issue while computing the eigendecomposition "
                "of the matrix." % (max_imag_abs / max_real_abs)
            )

        # warn about imaginary parts being removed
        if enable_warnings:
            warnings.warn(
                "There are imaginary parts in eigenvalues (%g "
                "of the maximum real part). Either the matrix is not"
                " PSD, or there was an issue while computing the "
                "eigendecomposition of the matrix. Only the real "
                "parts will be kept." % (max_imag_abs / max_real_abs),
                PositiveSpectrumWarning,
            )

    # Remove all imaginary parts (even if zero)
    lambdas = np.real(lambdas)

    # Check that there are no significant negative eigenvalues
    max_eig = lambdas.max()
    if max_eig < 0:
        raise ValueError(
            "All eigenvalues are negative (maximum is %g). "
            "Either the matrix is not PSD, or there was an "
            "issue while computing the eigendecomposition of "
            "the matrix." % max_eig
        )

    else:
        min_eig = lambdas.min()
        if (
            min_eig < -significant_neg_ratio * max_eig
            and min_eig < -significant_neg_value
        ):
            raise ValueError(
                "There are significant negative eigenvalues (%g"
                " of the maximum positive). Either the matrix is "
                "not PSD, or there was an issue while computing "
                "the eigendecomposition of the matrix." % (-min_eig / max_eig)
            )
        elif min_eig < 0:
            # Remove all negative values and warn about it
            if enable_warnings:
                warnings.warn(
                    "There are negative eigenvalues (%g of the "
                    "maximum positive). Either the matrix is not "
                    "PSD, or there was an issue while computing the"
                    " eigendecomposition of the matrix. Negative "
                    "eigenvalues will be replaced with 0." % (-min_eig / max_eig),
                    PositiveSpectrumWarning,
                )
            lambdas[lambdas < 0] = 0

    # Check for conditioning (small positive non-zeros)
    too_small_lambdas = (0 < lambdas) & (lambdas < small_pos_ratio * max_eig)
    if too_small_lambdas.any():
        if enable_warnings:
            warnings.warn(
                "Badly conditioned PSD matrix spectrum: the largest "
                "eigenvalue is more than %g times the smallest. "
                "Small eigenvalues will be replaced with 0."
                "" % (1 / small_pos_ratio),
                PositiveSpectrumWarning,
            )
        lambdas[too_small_lambdas] = 0

    return lambdas

