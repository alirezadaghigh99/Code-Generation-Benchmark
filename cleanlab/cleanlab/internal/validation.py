def labels_to_array(y: Union[LabelLike, np.generic]) -> np.ndarray:
    """Converts different types of label objects to 1D numpy array and checks their validity.

    Parameters
    ----------
    y : Union[LabelLike, np.generic]
        Labels to convert to 1D numpy array. Can be a list, numpy array, pandas Series, or pandas DataFrame.

    Returns
    -------
    labels_array : np.ndarray
        1D numpy array of labels.
    """
    if isinstance(y, pd.Series):
        y_series: np.ndarray = y.to_numpy()
        return y_series
    elif isinstance(y, pd.DataFrame):
        y_arr = y.values
        assert isinstance(y_arr, np.ndarray)
        if y_arr.shape[1] != 1:
            raise ValueError("labels must be one dimensional.")
        return y_arr.flatten()
    else:  # y is list, np.ndarray, or some other tuple-like object
        try:
            return np.asarray(y)
        except:
            raise ValueError(
                "List of labels must be convertable to 1D numpy array via: np.ndarray(labels)."
            )

