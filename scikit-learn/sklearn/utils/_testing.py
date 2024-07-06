def assert_allclose(
    actual, desired, rtol=None, atol=0.0, equal_nan=True, err_msg="", verbose=True
):
    """dtype-aware variant of numpy.testing.assert_allclose

    This variant introspects the least precise floating point dtype
    in the input argument and automatically sets the relative tolerance
    parameter to 1e-4 float32 and use 1e-7 otherwise (typically float64
    in scikit-learn).

    `atol` is always left to 0. by default. It should be adjusted manually
    to an assertion-specific value in case there are null values expected
    in `desired`.

    The aggregate tolerance is `atol + rtol * abs(desired)`.

    Parameters
    ----------
    actual : array_like
        Array obtained.
    desired : array_like
        Array desired.
    rtol : float, optional, default=None
        Relative tolerance.
        If None, it is set based on the provided arrays' dtypes.
    atol : float, optional, default=0.
        Absolute tolerance.
    equal_nan : bool, optional, default=True
        If True, NaNs will compare equal.
    err_msg : str, optional, default=''
        The error message to be printed in case of failure.
    verbose : bool, optional, default=True
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
        If actual and desired are not equal up to specified precision.

    See Also
    --------
    numpy.testing.assert_allclose

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.utils._testing import assert_allclose
    >>> x = [1e-5, 1e-3, 1e-1]
    >>> y = np.arccos(np.cos(x))
    >>> assert_allclose(x, y, rtol=1e-5, atol=0)
    >>> a = np.full(shape=10, fill_value=1e-5, dtype=np.float32)
    >>> assert_allclose(a, 1e-5)
    """
    dtypes = []

    actual, desired = np.asanyarray(actual), np.asanyarray(desired)
    dtypes = [actual.dtype, desired.dtype]

    if rtol is None:
        rtols = [1e-4 if dtype == np.float32 else 1e-7 for dtype in dtypes]
        rtol = max(rtols)

    np_assert_allclose(
        actual,
        desired,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        err_msg=err_msg,
        verbose=verbose,
    )

def ignore_warnings(obj=None, category=Warning):
    """Context manager and decorator to ignore warnings.

    Note: Using this (in both variants) will clear all warnings
    from all python modules loaded. In case you need to test
    cross-module-warning-logging, this is not your tool of choice.

    Parameters
    ----------
    obj : callable, default=None
        callable where you want to ignore the warnings.
    category : warning class, default=Warning
        The category to filter. If Warning, all categories will be muted.

    Examples
    --------
    >>> import warnings
    >>> from sklearn.utils._testing import ignore_warnings
    >>> with ignore_warnings():
    ...     warnings.warn('buhuhuhu')

    >>> def nasty_warn():
    ...     warnings.warn('buhuhuhu')
    ...     print(42)

    >>> ignore_warnings(nasty_warn)()
    42
    """
    if isinstance(obj, type) and issubclass(obj, Warning):
        # Avoid common pitfall of passing category as the first positional
        # argument which result in the test not being run
        warning_name = obj.__name__
        raise ValueError(
            "'obj' should be a callable where you want to ignore warnings. "
            "You passed a warning class instead: 'obj={warning_name}'. "
            "If you want to pass a warning class to ignore_warnings, "
            "you should use 'category={warning_name}'".format(warning_name=warning_name)
        )
    elif callable(obj):
        return _IgnoreWarnings(category=category)(obj)
    else:
        return _IgnoreWarnings(category=category)

def _convert_container(
    container,
    constructor_name,
    columns_name=None,
    dtype=None,
    minversion=None,
    categorical_feature_names=None,
):
    """Convert a given container to a specific array-like with a dtype.

    Parameters
    ----------
    container : array-like
        The container to convert.
    constructor_name : {"list", "tuple", "array", "sparse", "dataframe", \
            "series", "index", "slice", "sparse_csr", "sparse_csc", \
            "sparse_csr_array", "sparse_csc_array", "pyarrow", "polars", \
            "polars_series"}
        The type of the returned container.
    columns_name : index or array-like, default=None
        For pandas container supporting `columns_names`, it will affect
        specific names.
    dtype : dtype, default=None
        Force the dtype of the container. Does not apply to `"slice"`
        container.
    minversion : str, default=None
        Minimum version for package to install.
    categorical_feature_names : list of str, default=None
        List of column names to cast to categorical dtype.

    Returns
    -------
    converted_container
    """
    if constructor_name == "list":
        if dtype is None:
            return list(container)
        else:
            return np.asarray(container, dtype=dtype).tolist()
    elif constructor_name == "tuple":
        if dtype is None:
            return tuple(container)
        else:
            return tuple(np.asarray(container, dtype=dtype).tolist())
    elif constructor_name == "array":
        return np.asarray(container, dtype=dtype)
    elif constructor_name in ("pandas", "dataframe"):
        pd = pytest.importorskip("pandas", minversion=minversion)
        result = pd.DataFrame(container, columns=columns_name, dtype=dtype, copy=False)
        if categorical_feature_names is not None:
            for col_name in categorical_feature_names:
                result[col_name] = result[col_name].astype("category")
        return result
    elif constructor_name == "pyarrow":
        pa = pytest.importorskip("pyarrow", minversion=minversion)
        array = np.asarray(container)
        if columns_name is None:
            columns_name = [f"col{i}" for i in range(array.shape[1])]
        data = {name: array[:, i] for i, name in enumerate(columns_name)}
        result = pa.Table.from_pydict(data)
        if categorical_feature_names is not None:
            for col_idx, col_name in enumerate(result.column_names):
                if col_name in categorical_feature_names:
                    result = result.set_column(
                        col_idx, col_name, result.column(col_name).dictionary_encode()
                    )
        return result
    elif constructor_name == "polars":
        pl = pytest.importorskip("polars", minversion=minversion)
        result = pl.DataFrame(container, schema=columns_name, orient="row")
        if categorical_feature_names is not None:
            for col_name in categorical_feature_names:
                result = result.with_columns(pl.col(col_name).cast(pl.Categorical))
        return result
    elif constructor_name == "series":
        pd = pytest.importorskip("pandas", minversion=minversion)
        return pd.Series(container, dtype=dtype)
    elif constructor_name == "polars_series":
        pl = pytest.importorskip("polars", minversion=minversion)
        return pl.Series(values=container)
    elif constructor_name == "index":
        pd = pytest.importorskip("pandas", minversion=minversion)
        return pd.Index(container, dtype=dtype)
    elif constructor_name == "slice":
        return slice(container[0], container[1])
    elif "sparse" in constructor_name:
        if not sp.sparse.issparse(container):
            # For scipy >= 1.13, sparse array constructed from 1d array may be
            # 1d or raise an exception. To avoid this, we make sure that the
            # input container is 2d. For more details, see
            # https://github.com/scipy/scipy/pull/18530#issuecomment-1878005149
            container = np.atleast_2d(container)

        if "array" in constructor_name and sp_version < parse_version("1.8"):
            raise ValueError(
                f"{constructor_name} is only available with scipy>=1.8.0, got "
                f"{sp_version}"
            )
        if constructor_name in ("sparse", "sparse_csr"):
            # sparse and sparse_csr are equivalent for legacy reasons
            return sp.sparse.csr_matrix(container, dtype=dtype)
        elif constructor_name == "sparse_csr_array":
            return sp.sparse.csr_array(container, dtype=dtype)
        elif constructor_name == "sparse_csc":
            return sp.sparse.csc_matrix(container, dtype=dtype)
        elif constructor_name == "sparse_csc_array":
            return sp.sparse.csc_array(container, dtype=dtype)

def create_memmap_backed_data(data, mmap_mode="r", return_folder=False):
    """
    Parameters
    ----------
    data
    mmap_mode : str, default='r'
    return_folder :  bool, default=False
    """
    temp_folder = tempfile.mkdtemp(prefix="sklearn_testing_")
    atexit.register(functools.partial(_delete_folder, temp_folder, warn=True))
    filename = op.join(temp_folder, "data.pkl")
    joblib.dump(data, filename)
    memmap_backed_data = joblib.load(filename, mmap_mode=mmap_mode)
    result = (
        memmap_backed_data if not return_folder else (memmap_backed_data, temp_folder)
    )
    return result

def assert_run_python_script_without_output(source_code, pattern=".+", timeout=60):
    """Utility to check assertions in an independent Python subprocess.

    The script provided in the source code should return 0 and the stdtout +
    stderr should not match the pattern `pattern`.

    This is a port from cloudpickle https://github.com/cloudpipe/cloudpickle

    Parameters
    ----------
    source_code : str
        The Python source code to execute.
    pattern : str
        Pattern that the stdout + stderr should not match. By default, unless
        stdout + stderr are both empty, an error will be raised.
    timeout : int, default=60
        Time in seconds before timeout.
    """
    fd, source_file = tempfile.mkstemp(suffix="_src_test_sklearn.py")
    os.close(fd)
    try:
        with open(source_file, "wb") as f:
            f.write(source_code.encode("utf-8"))
        cmd = [sys.executable, source_file]
        cwd = op.normpath(op.join(op.dirname(sklearn.__file__), ".."))
        env = os.environ.copy()
        try:
            env["PYTHONPATH"] = os.pathsep.join([cwd, env["PYTHONPATH"]])
        except KeyError:
            env["PYTHONPATH"] = cwd
        kwargs = {"cwd": cwd, "stderr": STDOUT, "env": env}
        # If coverage is running, pass the config file to the subprocess
        coverage_rc = os.environ.get("COVERAGE_PROCESS_START")
        if coverage_rc:
            kwargs["env"]["COVERAGE_PROCESS_START"] = coverage_rc

        kwargs["timeout"] = timeout
        try:
            try:
                out = check_output(cmd, **kwargs)
            except CalledProcessError as e:
                raise RuntimeError(
                    "script errored with output:\n%s" % e.output.decode("utf-8")
                )

            out = out.decode("utf-8")
            if re.search(pattern, out):
                if pattern == ".+":
                    expectation = "Expected no output"
                else:
                    expectation = f"The output was not supposed to match {pattern!r}"

                message = f"{expectation}, got the following output instead: {out!r}"
                raise AssertionError(message)
        except TimeoutExpired as e:
            raise RuntimeError(
                "script timeout, output so far:\n%s" % e.output.decode("utf-8")
            )
    finally:
        os.unlink(source_file)

def assert_allclose_dense_sparse(x, y, rtol=1e-07, atol=1e-9, err_msg=""):
    """Assert allclose for sparse and dense data.

    Both x and y need to be either sparse or dense, they
    can't be mixed.

    Parameters
    ----------
    x : {array-like, sparse matrix}
        First array to compare.

    y : {array-like, sparse matrix}
        Second array to compare.

    rtol : float, default=1e-07
        relative tolerance; see numpy.allclose.

    atol : float, default=1e-9
        absolute tolerance; see numpy.allclose. Note that the default here is
        more tolerant than the default for numpy.testing.assert_allclose, where
        atol=0.

    err_msg : str, default=''
        Error message to raise.
    """
    if sp.sparse.issparse(x) and sp.sparse.issparse(y):
        x = x.tocsr()
        y = y.tocsr()
        x.sum_duplicates()
        y.sum_duplicates()
        assert_array_equal(x.indices, y.indices, err_msg=err_msg)
        assert_array_equal(x.indptr, y.indptr, err_msg=err_msg)
        assert_allclose(x.data, y.data, rtol=rtol, atol=atol, err_msg=err_msg)
    elif not sp.sparse.issparse(x) and not sp.sparse.issparse(y):
        # both dense
        assert_allclose(x, y, rtol=rtol, atol=atol, err_msg=err_msg)
    else:
        raise ValueError(
            "Can only compare two sparse matrices, not a sparse matrix and an array."
        )

def raises(expected_exc_type, match=None, may_pass=False, err_msg=None):
    """Context manager to ensure exceptions are raised within a code block.

    This is similar to and inspired from pytest.raises, but supports a few
    other cases.

    This is only intended to be used in estimator_checks.py where we don't
    want to use pytest. In the rest of the code base, just use pytest.raises
    instead.

    Parameters
    ----------
    excepted_exc_type : Exception or list of Exception
        The exception that should be raised by the block. If a list, the block
        should raise one of the exceptions.
    match : str or list of str, default=None
        A regex that the exception message should match. If a list, one of
        the entries must match. If None, match isn't enforced.
    may_pass : bool, default=False
        If True, the block is allowed to not raise an exception. Useful in
        cases where some estimators may support a feature but others must
        fail with an appropriate error message. By default, the context
        manager will raise an exception if the block does not raise an
        exception.
    err_msg : str, default=None
        If the context manager fails (e.g. the block fails to raise the
        proper exception, or fails to match), then an AssertionError is
        raised with this message. By default, an AssertionError is raised
        with a default error message (depends on the kind of failure). Use
        this to indicate how users should fix their estimators to pass the
        checks.

    Attributes
    ----------
    raised_and_matched : bool
        True if an exception was raised and a match was found, False otherwise.
    """
    return _Raises(expected_exc_type, match, may_pass, err_msg)

def _get_warnings_filters_info_list():
    @dataclass
    class WarningInfo:
        action: "warnings._ActionKind"
        message: str = ""
        category: type[Warning] = Warning

        def to_filterwarning_str(self):
            if self.category.__module__ == "builtins":
                category = self.category.__name__
            else:
                category = f"{self.category.__module__}.{self.category.__name__}"

            return f"{self.action}:{self.message}:{category}"

    return [
        WarningInfo("error", category=DeprecationWarning),
        WarningInfo("error", category=FutureWarning),
        WarningInfo("error", category=VisibleDeprecationWarning),
        # TODO: remove when pyamg > 5.0.1
        # Avoid a deprecation warning due pkg_resources usage in pyamg.
        WarningInfo(
            "ignore",
            message="pkg_resources is deprecated as an API",
            category=DeprecationWarning,
        ),
        WarningInfo(
            "ignore",
            message="Deprecated call to `pkg_resources",
            category=DeprecationWarning,
        ),
        # pytest-cov issue https://github.com/pytest-dev/pytest-cov/issues/557 not
        # fixed although it has been closed. https://github.com/pytest-dev/pytest-cov/pull/623
        # would probably fix it.
        WarningInfo(
            "ignore",
            message=(
                "The --rsyncdir command line argument and rsyncdirs config variable are"
                " deprecated"
            ),
            category=DeprecationWarning,
        ),
        # XXX: Easiest way to ignore pandas Pyarrow DeprecationWarning in the
        # short-term. See https://github.com/pandas-dev/pandas/issues/54466 for
        # more details.
        WarningInfo(
            "ignore",
            message=r"\s*Pyarrow will become a required dependency",
            category=DeprecationWarning,
        ),
        # warnings has been fixed from dateutil main but not released yet, see
        # https://github.com/dateutil/dateutil/issues/1314
        WarningInfo(
            "ignore",
            message="datetime.datetime.utcfromtimestamp",
            category=DeprecationWarning,
        ),
        # Python 3.12 warnings from joblib fixed in master but not released yet,
        # see https://github.com/joblib/joblib/pull/1518
        WarningInfo(
            "ignore", message="ast.Num is deprecated", category=DeprecationWarning
        ),
        WarningInfo(
            "ignore", message="Attribute n is deprecated", category=DeprecationWarning
        ),
        # Python 3.12 warnings from sphinx-gallery fixed in master but not
        # released yet, see
        # https://github.com/sphinx-gallery/sphinx-gallery/pull/1242
        WarningInfo(
            "ignore", message="ast.Str is deprecated", category=DeprecationWarning
        ),
        WarningInfo(
            "ignore", message="Attribute s is deprecated", category=DeprecationWarning
        ),
    ]

def check_docstring_parameters(func, doc=None, ignore=None):
    """Helper to check docstring.

    Parameters
    ----------
    func : callable
        The function object to test.
    doc : str, default=None
        Docstring if it is passed manually to the test.
    ignore : list, default=None
        Parameters to ignore.

    Returns
    -------
    incorrect : list
        A list of string describing the incorrect results.
    """
    from numpydoc import docscrape

    incorrect = []
    ignore = [] if ignore is None else ignore

    func_name = _get_func_name(func)
    if not func_name.startswith("sklearn.") or func_name.startswith(
        "sklearn.externals"
    ):
        return incorrect
    # Don't check docstring for property-functions
    if inspect.isdatadescriptor(func):
        return incorrect
    # Don't check docstring for setup / teardown pytest functions
    if func_name.split(".")[-1] in ("setup_module", "teardown_module"):
        return incorrect
    # Dont check estimator_checks module
    if func_name.split(".")[2] == "estimator_checks":
        return incorrect
    # Get the arguments from the function signature
    param_signature = list(filter(lambda x: x not in ignore, _get_args(func)))
    # drop self
    if len(param_signature) > 0 and param_signature[0] == "self":
        param_signature.remove("self")

    # Analyze function's docstring
    if doc is None:
        records = []
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("error", UserWarning)
            try:
                doc = docscrape.FunctionDoc(func)
            except UserWarning as exp:
                if "potentially wrong underline length" in str(exp):
                    # Catch warning raised as of numpydoc 1.2 when
                    # the underline length for a section of a docstring
                    # is not consistent.
                    message = str(exp).split("\n")[:3]
                    incorrect += [f"In function: {func_name}"] + message
                    return incorrect
                records.append(str(exp))
            except Exception as exp:
                incorrect += [func_name + " parsing error: " + str(exp)]
                return incorrect
        if len(records):
            raise RuntimeError("Error for %s:\n%s" % (func_name, records[0]))

    param_docs = []
    for name, type_definition, param_doc in doc["Parameters"]:
        # Type hints are empty only if parameter name ended with :
        if not type_definition.strip():
            if ":" in name and name[: name.index(":")][-1:].strip():
                incorrect += [
                    func_name
                    + " There was no space between the param name and colon (%r)" % name
                ]
            elif name.rstrip().endswith(":"):
                incorrect += [
                    func_name
                    + " Parameter %r has an empty type spec. Remove the colon"
                    % (name.lstrip())
                ]

        # Create a list of parameters to compare with the parameters gotten
        # from the func signature
        if "*" not in name:
            param_docs.append(name.split(":")[0].strip("` "))

    # If one of the docstring's parameters had an error then return that
    # incorrect message
    if len(incorrect) > 0:
        return incorrect

    # Remove the parameters that should be ignored from list
    param_docs = list(filter(lambda x: x not in ignore, param_docs))

    # The following is derived from pytest, Copyright (c) 2004-2017 Holger
    # Krekel and others, Licensed under MIT License. See
    # https://github.com/pytest-dev/pytest

    message = []
    for i in range(min(len(param_docs), len(param_signature))):
        if param_signature[i] != param_docs[i]:
            message += [
                "There's a parameter name mismatch in function"
                " docstring w.r.t. function signature, at index %s"
                " diff: %r != %r" % (i, param_signature[i], param_docs[i])
            ]
            break
    if len(param_signature) > len(param_docs):
        message += [
            "Parameters in function docstring have less items w.r.t."
            " function signature, first missing item: %s"
            % param_signature[len(param_docs)]
        ]

    elif len(param_signature) < len(param_docs):
        message += [
            "Parameters in function docstring have more items w.r.t."
            " function signature, first extra item: %s"
            % param_docs[len(param_signature)]
        ]

    # If there wasn't any difference in the parameters themselves between
    # docstring and signature including having the same length then return
    # empty list
    if len(message) == 0:
        return []

    import difflib
    import pprint

    param_docs_formatted = pprint.pformat(param_docs).splitlines()
    param_signature_formatted = pprint.pformat(param_signature).splitlines()

    message += ["Full diff:"]

    message.extend(
        line.strip()
        for line in difflib.ndiff(param_signature_formatted, param_docs_formatted)
    )

    incorrect.extend(message)

    # Prepend function name
    incorrect = ["In function: " + func_name] + incorrect

    return incorrect

