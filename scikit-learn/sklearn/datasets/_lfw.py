def fetch_lfw_people(
    *,
    data_home=None,
    funneled=True,
    resize=0.5,
    min_faces_per_person=0,
    color=False,
    slice_=(slice(70, 195), slice(78, 172)),
    download_if_missing=True,
    return_X_y=False,
    n_retries=3,
    delay=1.0,
):
    """Load the Labeled Faces in the Wild (LFW) people dataset \
(classification).

    Download it if necessary.

    =================   =======================
    Classes                                5749
    Samples total                         13233
    Dimensionality                         5828
    Features            real, between 0 and 255
    =================   =======================

    For a usage example of this dataset, see
    :ref:`sphx_glr_auto_examples_applications_plot_face_recognition.py`.

    Read more in the :ref:`User Guide <labeled_faces_in_the_wild_dataset>`.

    Parameters
    ----------
    data_home : str or path-like, default=None
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    funneled : bool, default=True
        Download and use the funneled variant of the dataset.

    resize : float or None, default=0.5
        Ratio used to resize the each face picture. If `None`, no resizing is
        performed.

    min_faces_per_person : int, default=None
        The extracted dataset will only retain pictures of people that have at
        least `min_faces_per_person` different pictures.

    color : bool, default=False
        Keep the 3 RGB channels instead of averaging them to a single
        gray level channel. If color is True the shape of the data has
        one more dimension than the shape with color = False.

    slice_ : tuple of slice, default=(slice(70, 195), slice(78, 172))
        Provide a custom 2D slice (height, width) to extract the
        'interesting' part of the jpeg files and avoid use statistical
        correlation from the background.

    download_if_missing : bool, default=True
        If False, raise an OSError if the data is not locally available
        instead of trying to download the data from the source site.

    return_X_y : bool, default=False
        If True, returns ``(dataset.data, dataset.target)`` instead of a Bunch
        object. See below for more information about the `dataset.data` and
        `dataset.target` object.

        .. versionadded:: 0.20

    n_retries : int, default=3
        Number of retries when HTTP errors are encountered.

        .. versionadded:: 1.5

    delay : float, default=1.0
        Number of seconds between retries.

        .. versionadded:: 1.5

    Returns
    -------
    dataset : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : numpy array of shape (13233, 2914)
            Each row corresponds to a ravelled face image
            of original size 62 x 47 pixels.
            Changing the ``slice_`` or resize parameters will change the
            shape of the output.
        images : numpy array of shape (13233, 62, 47)
            Each row is a face image corresponding to one of the 5749 people in
            the dataset. Changing the ``slice_``
            or resize parameters will change the shape of the output.
        target : numpy array of shape (13233,)
            Labels associated to each face image.
            Those labels range from 0-5748 and correspond to the person IDs.
        target_names : numpy array of shape (5749,)
            Names of all persons in the dataset.
            Position in array corresponds to the person ID in the target array.
        DESCR : str
            Description of the Labeled Faces in the Wild (LFW) dataset.

    (data, target) : tuple if ``return_X_y`` is True
        A tuple of two ndarray. The first containing a 2D array of
        shape (n_samples, n_features) with each row representing one
        sample and each column representing the features. The second
        ndarray of shape (n_samples,) containing the target samples.

        .. versionadded:: 0.20

    Examples
    --------
    >>> from sklearn.datasets import fetch_lfw_people
    >>> lfw_people = fetch_lfw_people()
    >>> lfw_people.data.shape
    (13233, 2914)
    >>> lfw_people.target.shape
    (13233,)
    >>> for name in lfw_people.target_names[:5]:
    ...    print(name)
    AJ Cook
    AJ Lamas
    Aaron Eckhart
    Aaron Guiel
    Aaron Patterson
    """
    lfw_home, data_folder_path = _check_fetch_lfw(
        data_home=data_home,
        funneled=funneled,
        download_if_missing=download_if_missing,
        n_retries=n_retries,
        delay=delay,
    )
    logger.debug("Loading LFW people faces from %s", lfw_home)

    # wrap the loader in a memoizing function that will return memmaped data
    # arrays for optimal memory usage
    m = Memory(location=lfw_home, compress=6, verbose=0)
    load_func = m.cache(_fetch_lfw_people)

    # load and memoize the pairs as np arrays
    faces, target, target_names = load_func(
        data_folder_path,
        resize=resize,
        min_faces_per_person=min_faces_per_person,
        color=color,
        slice_=slice_,
    )

    X = faces.reshape(len(faces), -1)

    fdescr = load_descr("lfw.rst")

    if return_X_y:
        return X, target

    # pack the results as a Bunch instance
    return Bunch(
        data=X, images=faces, target=target, target_names=target_names, DESCR=fdescr
    )

