def shape_sanity_check(image: np.ndarray, ddf: np.ndarray):
    """
    Verify image and ddf shapes are consistent and correct.

    :param image: a numpy array of shape (m_dim1, m_dim2, m_dim3)
        or (m_dim1, m_dim2, m_dim3, ch)
    :param ddf: a numpy array of shape (f_dim1, f_dim2, f_dim3, 3)
    """
    if len(image.shape) not in [3, 4]:
        raise ValueError(
            f"image shape must be (m_dim1, m_dim2, m_dim3) "
            f"or (m_dim1, m_dim2, m_dim3, ch),"
            f" got {image.shape}"
        )
    if not (len(ddf.shape) == 4 and ddf.shape[-1] == 3):
        raise ValueError(
            f"ddf shape must be (f_dim1, f_dim2, f_dim3, 3), got {ddf.shape}"
        )

