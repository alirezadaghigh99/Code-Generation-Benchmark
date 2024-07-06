def extract_patches_2d(image, patch_size, *, max_patches=None, random_state=None):
    """Reshape a 2D image into a collection of patches.

    The resulting patches are allocated in a dedicated array.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    image : ndarray of shape (image_height, image_width) or \
        (image_height, image_width, n_channels)
        The original image data. For color images, the last dimension specifies
        the channel: a RGB image would have `n_channels=3`.

    patch_size : tuple of int (patch_height, patch_width)
        The dimensions of one patch.

    max_patches : int or float, default=None
        The maximum number of patches to extract. If `max_patches` is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches. If `max_patches` is None it corresponds to the total number
        of patches that can be extracted.

    random_state : int, RandomState instance, default=None
        Determines the random number generator used for random sampling when
        `max_patches` is not None. Use an int to make the randomness
        deterministic.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    patches : array of shape (n_patches, patch_height, patch_width) or \
        (n_patches, patch_height, patch_width, n_channels)
        The collection of patches extracted from the image, where `n_patches`
        is either `max_patches` or the total number of patches that can be
        extracted.

    Examples
    --------
    >>> from sklearn.datasets import load_sample_image
    >>> from sklearn.feature_extraction import image
    >>> # Use the array data from the first image in this dataset:
    >>> one_image = load_sample_image("china.jpg")
    >>> print('Image shape: {}'.format(one_image.shape))
    Image shape: (427, 640, 3)
    >>> patches = image.extract_patches_2d(one_image, (2, 2))
    >>> print('Patches shape: {}'.format(patches.shape))
    Patches shape: (272214, 2, 2, 3)
    >>> # Here are just two of these patches:
    >>> print(patches[1])
    [[[174 201 231]
      [174 201 231]]
     [[173 200 230]
      [173 200 230]]]
    >>> print(patches[800])
    [[[187 214 243]
      [188 215 244]]
     [[187 214 243]
      [188 215 244]]]
    """
    i_h, i_w = image.shape[:2]
    p_h, p_w = patch_size

    if p_h > i_h:
        raise ValueError(
            "Height of the patch should be less than the height of the image."
        )

    if p_w > i_w:
        raise ValueError(
            "Width of the patch should be less than the width of the image."
        )

    image = check_array(image, allow_nd=True)
    image = image.reshape((i_h, i_w, -1))
    n_colors = image.shape[-1]

    extracted_patches = _extract_patches(
        image, patch_shape=(p_h, p_w, n_colors), extraction_step=1
    )

    n_patches = _compute_n_patches(i_h, i_w, p_h, p_w, max_patches)
    if max_patches:
        rng = check_random_state(random_state)
        i_s = rng.randint(i_h - p_h + 1, size=n_patches)
        j_s = rng.randint(i_w - p_w + 1, size=n_patches)
        patches = extracted_patches[i_s, j_s, 0]
    else:
        patches = extracted_patches

    patches = patches.reshape(-1, p_h, p_w, n_colors)
    # remove the color dimension if useless
    if patches.shape[-1] == 1:
        return patches.reshape((n_patches, p_h, p_w))
    else:
        return patches

def grid_to_graph(
    n_x, n_y, n_z=1, *, mask=None, return_as=sparse.coo_matrix, dtype=int
):
    """Graph of the pixel-to-pixel connections.

    Edges exist if 2 voxels are connected.

    Parameters
    ----------
    n_x : int
        Dimension in x axis.
    n_y : int
        Dimension in y axis.
    n_z : int, default=1
        Dimension in z axis.
    mask : ndarray of shape (n_x, n_y, n_z), dtype=bool, default=None
        An optional mask of the image, to consider only part of the
        pixels.
    return_as : np.ndarray or a sparse matrix class, \
            default=sparse.coo_matrix
        The class to use to build the returned adjacency matrix.
    dtype : dtype, default=int
        The data of the returned sparse matrix. By default it is int.

    Returns
    -------
    graph : np.ndarray or a sparse matrix class
        The computed adjacency matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.feature_extraction.image import grid_to_graph
    >>> shape_img = (4, 4, 1)
    >>> mask = np.zeros(shape=shape_img, dtype=bool)
    >>> mask[[1, 2], [1, 2], :] = True
    >>> graph = grid_to_graph(*shape_img, mask=mask)
    >>> print(graph)
      (0, 0)    1
      (1, 1)    1
    """
    return _to_graph(n_x, n_y, n_z, mask=mask, return_as=return_as, dtype=dtype)

def img_to_graph(img, *, mask=None, return_as=sparse.coo_matrix, dtype=None):
    """Graph of the pixel-to-pixel gradient connections.

    Edges are weighted with the gradient values.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    img : array-like of shape (height, width) or (height, width, channel)
        2D or 3D image.
    mask : ndarray of shape (height, width) or \
            (height, width, channel), dtype=bool, default=None
        An optional mask of the image, to consider only part of the
        pixels.
    return_as : np.ndarray or a sparse matrix class, \
            default=sparse.coo_matrix
        The class to use to build the returned adjacency matrix.
    dtype : dtype, default=None
        The data of the returned sparse matrix. By default it is the
        dtype of img.

    Returns
    -------
    graph : ndarray or a sparse matrix class
        The computed adjacency matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.feature_extraction.image import img_to_graph
    >>> img = np.array([[0, 0], [0, 1]])
    >>> img_to_graph(img, return_as=np.ndarray)
    array([[0, 0, 0, 0],
           [0, 0, 0, 1],
           [0, 0, 0, 1],
           [0, 1, 1, 1]])
    """
    img = np.atleast_3d(img)
    n_x, n_y, n_z = img.shape
    return _to_graph(n_x, n_y, n_z, mask, img, return_as, dtype)

def _extract_patches(arr, patch_shape=8, extraction_step=1):
    """Extracts patches of any n-dimensional array in place using strides.

    Given an n-dimensional array it will return a 2n-dimensional array with
    the first n dimensions indexing patch position and the last n indexing
    the patch content. This operation is immediate (O(1)). A reshape
    performed on the first n dimensions will cause numpy to copy data, leading
    to a list of extracted patches.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    arr : ndarray
        n-dimensional array of which patches are to be extracted

    patch_shape : int or tuple of length arr.ndim.default=8
        Indicates the shape of the patches to be extracted. If an
        integer is given, the shape will be a hypercube of
        sidelength given by its value.

    extraction_step : int or tuple of length arr.ndim, default=1
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.


    Returns
    -------
    patches : strided ndarray
        2n-dimensional array indexing patches on first n dimensions and
        containing patches on the last n dimensions. These dimensions
        are fake, but this way no data is copied. A simple reshape invokes
        a copying operation to obtain a list of patches:
        result.reshape([-1] + list(patch_shape))
    """

    arr_ndim = arr.ndim

    if isinstance(patch_shape, Number):
        patch_shape = tuple([patch_shape] * arr_ndim)
    if isinstance(extraction_step, Number):
        extraction_step = tuple([extraction_step] * arr_ndim)

    patch_strides = arr.strides

    slices = tuple(slice(None, None, st) for st in extraction_step)
    indexing_strides = arr[slices].strides

    patch_indices_shape = (
        (np.array(arr.shape) - np.array(patch_shape)) // np.array(extraction_step)
    ) + 1

    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))

    patches = as_strided(arr, shape=shape, strides=strides)
    return patches
