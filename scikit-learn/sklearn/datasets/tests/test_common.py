def check_return_X_y(bunch, dataset_func):
    X_y_tuple = dataset_func(return_X_y=True)
    assert isinstance(X_y_tuple, tuple)
    assert X_y_tuple[0].shape == bunch.data.shape
    assert X_y_tuple[1].shape == bunch.target.shape

def check_as_frame(
    bunch, dataset_func, expected_data_dtype=None, expected_target_dtype=None
):
    pd = pytest.importorskip("pandas")
    frame_bunch = dataset_func(as_frame=True)
    assert hasattr(frame_bunch, "frame")
    assert isinstance(frame_bunch.frame, pd.DataFrame)
    assert isinstance(frame_bunch.data, pd.DataFrame)
    assert frame_bunch.data.shape == bunch.data.shape
    if frame_bunch.target.ndim > 1:
        assert isinstance(frame_bunch.target, pd.DataFrame)
    else:
        assert isinstance(frame_bunch.target, pd.Series)
    assert frame_bunch.target.shape[0] == bunch.target.shape[0]
    if expected_data_dtype is not None:
        assert np.all(frame_bunch.data.dtypes == expected_data_dtype)
    if expected_target_dtype is not None:
        assert np.all(frame_bunch.target.dtypes == expected_target_dtype)

    # Test for return_X_y and as_frame=True
    frame_X, frame_y = dataset_func(as_frame=True, return_X_y=True)
    assert isinstance(frame_X, pd.DataFrame)
    if frame_y.ndim > 1:
        assert isinstance(frame_X, pd.DataFrame)
    else:
        assert isinstance(frame_y, pd.Series)

def check_pandas_dependency_message(fetch_func):
    try:
        import pandas  # noqa

        pytest.skip("This test requires pandas to not be installed")
    except ImportError:
        # Check that pandas is imported lazily and that an informative error
        # message is raised when pandas is missing:
        name = fetch_func.__name__
        expected_msg = f"{name} with as_frame=True requires pandas"
        with pytest.raises(ImportError, match=expected_msg):
            fetch_func(as_frame=True)

