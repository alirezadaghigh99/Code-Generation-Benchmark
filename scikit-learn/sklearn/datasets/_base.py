def load_iris(*, return_X_y=False, as_frame=False):
    """Load and return the iris dataset (classification).

    The iris dataset is a classic and very easy multi-class classification
    dataset.

    =================   ==============
    Classes                          3
    Samples per class               50
    Samples total                  150
    Dimensionality                   4
    Features            real, positive
    =================   ==============

    Read more in the :ref:`User Guide <iris_dataset>`.

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object. See
        below for more information about the `data` and `target` object.

        .. versionadded:: 0.18

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

        .. versionadded:: 0.23

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : {ndarray, dataframe} of shape (150, 4)
            The data matrix. If `as_frame=True`, `data` will be a pandas
            DataFrame.
        target: {ndarray, Series} of shape (150,)
            The classification target. If `as_frame=True`, `target` will be
            a pandas Series.
        feature_names: list
            The names of the dataset columns.
        target_names: list
            The names of target classes.
        frame: DataFrame of shape (150, 5)
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.

            .. versionadded:: 0.23
        DESCR: str
            The full description of the dataset.
        filename: str
            The path to the location of the data.

            .. versionadded:: 0.20

    (data, target) : tuple if ``return_X_y`` is True
        A tuple of two ndarray. The first containing a 2D array of shape
        (n_samples, n_features) with each row representing one sample and
        each column representing the features. The second ndarray of shape
        (n_samples,) containing the target samples.

        .. versionadded:: 0.18

    Notes
    -----
        .. versionchanged:: 0.20
            Fixed two wrong data points according to Fisher's paper.
            The new version is the same as in R, but not as in the UCI
            Machine Learning Repository.

    Examples
    --------
    Let's say you are interested in the samples 10, 25, and 50, and want to
    know their class name.

    >>> from sklearn.datasets import load_iris
    >>> data = load_iris()
    >>> data.target[[10, 25, 50]]
    array([0, 0, 1])
    >>> list(data.target_names)
    ['setosa', 'versicolor', 'virginica']

    See :ref:`sphx_glr_auto_examples_datasets_plot_iris_dataset.py` for a more
    detailed example of how to work with the iris dataset.
    """
    data_file_name = "iris.csv"
    data, target, target_names, fdescr = load_csv_data(
        data_file_name=data_file_name, descr_file_name="iris.rst"
    )

    feature_names = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]

    frame = None
    target_columns = [
        "target",
    ]
    if as_frame:
        frame, data, target = _convert_data_dataframe(
            "load_iris", data, target, feature_names, target_columns
        )

    if return_X_y:
        return data, target

    return Bunch(
        data=data,
        target=target,
        frame=frame,
        target_names=target_names,
        DESCR=fdescr,
        feature_names=feature_names,
        filename=data_file_name,
        data_module=DATA_MODULE,
    )

def load_linnerud(*, return_X_y=False, as_frame=False):
    """Load and return the physical exercise Linnerud dataset.

    This dataset is suitable for multi-output regression tasks.

    ==============   ============================
    Samples total    20
    Dimensionality   3 (for both data and target)
    Features         integer
    Targets          integer
    ==============   ============================

    Read more in the :ref:`User Guide <linnerrud_dataset>`.

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

        .. versionadded:: 0.18

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric, string or categorical). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

        .. versionadded:: 0.23

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : {ndarray, dataframe} of shape (20, 3)
            The data matrix. If `as_frame=True`, `data` will be a pandas
            DataFrame.
        target: {ndarray, dataframe} of shape (20, 3)
            The regression targets. If `as_frame=True`, `target` will be
            a pandas DataFrame.
        feature_names: list
            The names of the dataset columns.
        target_names: list
            The names of the target columns.
        frame: DataFrame of shape (20, 6)
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.

            .. versionadded:: 0.23
        DESCR: str
            The full description of the dataset.
        data_filename: str
            The path to the location of the data.
        target_filename: str
            The path to the location of the target.

            .. versionadded:: 0.20

    (data, target) : tuple if ``return_X_y`` is True
        Returns a tuple of two ndarrays or dataframe of shape
        `(20, 3)`. Each row represents one sample and each column represents the
        features in `X` and a target in `y` of a given sample.

        .. versionadded:: 0.18

    Examples
    --------
    >>> from sklearn.datasets import load_linnerud
    >>> linnerud = load_linnerud()
    >>> linnerud.data.shape
    (20, 3)
    >>> linnerud.target.shape
    (20, 3)
    """
    data_filename = "linnerud_exercise.csv"
    target_filename = "linnerud_physiological.csv"

    data_module_path = resources.files(DATA_MODULE)
    # Read header and data
    data_path = data_module_path / data_filename
    with data_path.open("r", encoding="utf-8") as f:
        header_exercise = f.readline().split()
        f.seek(0)  # reset file obj
        data_exercise = np.loadtxt(f, skiprows=1)

    target_path = data_module_path / target_filename
    with target_path.open("r", encoding="utf-8") as f:
        header_physiological = f.readline().split()
        f.seek(0)  # reset file obj
        data_physiological = np.loadtxt(f, skiprows=1)

    fdescr = load_descr("linnerud.rst")

    frame = None
    if as_frame:
        (frame, data_exercise, data_physiological) = _convert_data_dataframe(
            "load_linnerud",
            data_exercise,
            data_physiological,
            header_exercise,
            header_physiological,
        )
    if return_X_y:
        return data_exercise, data_physiological

    return Bunch(
        data=data_exercise,
        feature_names=header_exercise,
        target=data_physiological,
        target_names=header_physiological,
        frame=frame,
        DESCR=fdescr,
        data_filename=data_filename,
        target_filename=target_filename,
        data_module=DATA_MODULE,
    )

def load_files(
    container_path,
    *,
    description=None,
    categories=None,
    load_content=True,
    shuffle=True,
    encoding=None,
    decode_error="strict",
    random_state=0,
    allowed_extensions=None,
):
    """Load text files with categories as subfolder names.

    Individual samples are assumed to be files stored a two levels folder
    structure such as the following:

        container_folder/
            category_1_folder/
                file_1.txt
                file_2.txt
                ...
                file_42.txt
            category_2_folder/
                file_43.txt
                file_44.txt
                ...

    The folder names are used as supervised signal label names. The individual
    file names are not important.

    This function does not try to extract features into a numpy array or scipy
    sparse matrix. In addition, if load_content is false it does not try to
    load the files in memory.

    To use text files in a scikit-learn classification or clustering algorithm,
    you will need to use the :mod:`~sklearn.feature_extraction.text` module to
    build a feature extraction transformer that suits your problem.

    If you set load_content=True, you should also specify the encoding of the
    text using the 'encoding' parameter. For many modern text files, 'utf-8'
    will be the correct encoding. If you leave encoding equal to None, then the
    content will be made of bytes instead of Unicode, and you will not be able
    to use most functions in :mod:`~sklearn.feature_extraction.text`.

    Similar feature extractors should be built for other kind of unstructured
    data input such as images, audio, video, ...

    If you want files with a specific file extension (e.g. `.txt`) then you
    can pass a list of those file extensions to `allowed_extensions`.

    Read more in the :ref:`User Guide <datasets>`.

    Parameters
    ----------
    container_path : str
        Path to the main folder holding one subfolder per category.

    description : str, default=None
        A paragraph describing the characteristic of the dataset: its source,
        reference, etc.

    categories : list of str, default=None
        If None (default), load all the categories. If not None, list of
        category names to load (other categories ignored).

    load_content : bool, default=True
        Whether to load or not the content of the different files. If true a
        'data' attribute containing the text information is present in the data
        structure returned. If not, a filenames attribute gives the path to the
        files.

    shuffle : bool, default=True
        Whether or not to shuffle the data: might be important for models that
        make the assumption that the samples are independent and identically
        distributed (i.i.d.), such as stochastic gradient descent.

    encoding : str, default=None
        If None, do not try to decode the content of the files (e.g. for images
        or other non-text content). If not None, encoding to use to decode text
        files to Unicode if load_content is True.

    decode_error : {'strict', 'ignore', 'replace'}, default='strict'
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. Passed as keyword
        argument 'errors' to bytes.decode.

    random_state : int, RandomState instance or None, default=0
        Determines random number generation for dataset shuffling. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    allowed_extensions : list of str, default=None
        List of desired file extensions to filter the files to be loaded.

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : list of str
            Only present when `load_content=True`.
            The raw text data to learn.
        target : ndarray
            The target labels (integer index).
        target_names : list
            The names of target classes.
        DESCR : str
            The full description of the dataset.
        filenames: ndarray
            The filenames holding the dataset.

    Examples
    --------
    >>> from sklearn.datasets import load_files
    >>> container_path = "./"
    >>> load_files(container_path)  # doctest: +SKIP
    """

    target = []
    target_names = []
    filenames = []

    folders = [
        f for f in sorted(listdir(container_path)) if isdir(join(container_path, f))
    ]

    if categories is not None:
        folders = [f for f in folders if f in categories]

    if allowed_extensions is not None:
        allowed_extensions = frozenset(allowed_extensions)

    for label, folder in enumerate(folders):
        target_names.append(folder)
        folder_path = join(container_path, folder)
        files = sorted(listdir(folder_path))
        if allowed_extensions is not None:
            documents = [
                join(folder_path, file)
                for file in files
                if os.path.splitext(file)[1] in allowed_extensions
            ]
        else:
            documents = [join(folder_path, file) for file in files]
        target.extend(len(documents) * [label])
        filenames.extend(documents)

    # convert to array for fancy indexing
    filenames = np.array(filenames)
    target = np.array(target)

    if shuffle:
        random_state = check_random_state(random_state)
        indices = np.arange(filenames.shape[0])
        random_state.shuffle(indices)
        filenames = filenames[indices]
        target = target[indices]

    if load_content:
        data = []
        for filename in filenames:
            data.append(Path(filename).read_bytes())
        if encoding is not None:
            data = [d.decode(encoding, decode_error) for d in data]
        return Bunch(
            data=data,
            filenames=filenames,
            target_names=target_names,
            target=target,
            DESCR=description,
        )

    return Bunch(
        filenames=filenames, target_names=target_names, target=target, DESCR=description
    )

def load_csv_data(
    data_file_name,
    *,
    data_module=DATA_MODULE,
    descr_file_name=None,
    descr_module=DESCR_MODULE,
    encoding="utf-8",
):
    """Loads `data_file_name` from `data_module with `importlib.resources`.

    Parameters
    ----------
    data_file_name : str
        Name of csv file to be loaded from `data_module/data_file_name`.
        For example `'wine_data.csv'`.

    data_module : str or module, default='sklearn.datasets.data'
        Module where data lives. The default is `'sklearn.datasets.data'`.

    descr_file_name : str, default=None
        Name of rst file to be loaded from `descr_module/descr_file_name`.
        For example `'wine_data.rst'`. See also :func:`load_descr`.
        If not None, also returns the corresponding description of
        the dataset.

    descr_module : str or module, default='sklearn.datasets.descr'
        Module where `descr_file_name` lives. See also :func:`load_descr`.
        The default is `'sklearn.datasets.descr'`.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features)
        A 2D array with each row representing one sample and each column
        representing the features of a given sample.

    target : ndarry of shape (n_samples,)
        A 1D array holding target variables for all the samples in `data`.
        For example target[0] is the target variable for data[0].

    target_names : ndarry of shape (n_samples,)
        A 1D array containing the names of the classifications. For example
        target_names[0] is the name of the target[0] class.

    descr : str, optional
        Description of the dataset (the content of `descr_file_name`).
        Only returned if `descr_file_name` is not None.

    encoding : str, optional
        Text encoding of the CSV file.

        .. versionadded:: 1.4
    """
    data_path = resources.files(data_module) / data_file_name
    with data_path.open("r", encoding="utf-8") as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        target_names = np.array(temp[2:])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=np.float64)
            target[i] = np.asarray(ir[-1], dtype=int)

    if descr_file_name is None:
        return data, target, target_names
    else:
        assert descr_module is not None
        descr = load_descr(descr_module=descr_module, descr_file_name=descr_file_name)
        return data, target, target_names, descr

def load_gzip_compressed_csv_data(
    data_file_name,
    *,
    data_module=DATA_MODULE,
    descr_file_name=None,
    descr_module=DESCR_MODULE,
    encoding="utf-8",
    **kwargs,
):
    """Loads gzip-compressed with `importlib.resources`.

    1) Open resource file with `importlib.resources.open_binary`
    2) Decompress file obj with `gzip.open`
    3) Load decompressed data with `np.loadtxt`

    Parameters
    ----------
    data_file_name : str
        Name of gzip-compressed csv file  (`'*.csv.gz'`) to be loaded from
        `data_module/data_file_name`. For example `'diabetes_data.csv.gz'`.

    data_module : str or module, default='sklearn.datasets.data'
        Module where data lives. The default is `'sklearn.datasets.data'`.

    descr_file_name : str, default=None
        Name of rst file to be loaded from `descr_module/descr_file_name`.
        For example `'wine_data.rst'`. See also :func:`load_descr`.
        If not None, also returns the corresponding description of
        the dataset.

    descr_module : str or module, default='sklearn.datasets.descr'
        Module where `descr_file_name` lives. See also :func:`load_descr`.
        The default  is `'sklearn.datasets.descr'`.

    encoding : str, default="utf-8"
        Name of the encoding that the gzip-decompressed file will be
        decoded with. The default is 'utf-8'.

    **kwargs : dict, optional
        Keyword arguments to be passed to `np.loadtxt`;
        e.g. delimiter=','.

    Returns
    -------
    data : ndarray of shape (n_samples, n_features)
        A 2D array with each row representing one sample and each column
        representing the features and/or target of a given sample.

    descr : str, optional
        Description of the dataset (the content of `descr_file_name`).
        Only returned if `descr_file_name` is not None.
    """
    data_path = resources.files(data_module) / data_file_name
    with data_path.open("rb") as compressed_file:
        compressed_file = gzip.open(compressed_file, mode="rt", encoding=encoding)
        data = np.loadtxt(compressed_file, **kwargs)

    if descr_file_name is None:
        return data
    else:
        assert descr_module is not None
        descr = load_descr(descr_module=descr_module, descr_file_name=descr_file_name)
        return data, descr

def load_diabetes(*, return_X_y=False, as_frame=False, scaled=True):
    """Load and return the diabetes dataset (regression).

    ==============   ==================
    Samples total    442
    Dimensionality   10
    Features         real, -.2 < x < .2
    Targets          integer 25 - 346
    ==============   ==================

    .. note::
       The meaning of each feature (i.e. `feature_names`) might be unclear
       (especially for `ltg`) as the documentation of the original dataset is
       not explicit. We provide information that seems correct in regard with
       the scientific literature in this field of research.

    Read more in the :ref:`User Guide <diabetes_dataset>`.

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

        .. versionadded:: 0.18

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

        .. versionadded:: 0.23

    scaled : bool, default=True
        If True, the feature variables are mean centered and scaled by the
        standard deviation times the square root of `n_samples`.
        If False, raw data is returned for the feature variables.

        .. versionadded:: 1.1

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : {ndarray, dataframe} of shape (442, 10)
            The data matrix. If `as_frame=True`, `data` will be a pandas
            DataFrame.
        target: {ndarray, Series} of shape (442,)
            The regression target. If `as_frame=True`, `target` will be
            a pandas Series.
        feature_names: list
            The names of the dataset columns.
        frame: DataFrame of shape (442, 11)
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.

            .. versionadded:: 0.23
        DESCR: str
            The full description of the dataset.
        data_filename: str
            The path to the location of the data.
        target_filename: str
            The path to the location of the target.

    (data, target) : tuple if ``return_X_y`` is True
        Returns a tuple of two ndarray of shape (n_samples, n_features)
        A 2D array with each row representing one sample and each column
        representing the features and/or target of a given sample.

        .. versionadded:: 0.18

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> diabetes = load_diabetes()
    >>> diabetes.target[:3]
    array([151.,  75., 141.])
    >>> diabetes.data.shape
    (442, 10)
    """
    data_filename = "diabetes_data_raw.csv.gz"
    target_filename = "diabetes_target.csv.gz"
    data = load_gzip_compressed_csv_data(data_filename)
    target = load_gzip_compressed_csv_data(target_filename)

    if scaled:
        data = scale(data, copy=False)
        data /= data.shape[0] ** 0.5

    fdescr = load_descr("diabetes.rst")

    feature_names = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]

    frame = None
    target_columns = [
        "target",
    ]
    if as_frame:
        frame, data, target = _convert_data_dataframe(
            "load_diabetes", data, target, feature_names, target_columns
        )

    if return_X_y:
        return data, target

    return Bunch(
        data=data,
        target=target,
        frame=frame,
        DESCR=fdescr,
        feature_names=feature_names,
        data_filename=data_filename,
        target_filename=target_filename,
        data_module=DATA_MODULE,
    )

