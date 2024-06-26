def get_naive_residuals(target_data: pd.DataFrame, m: int = 1) -> Tuple[List, float]:
    """
    Computes forecasting residuals for the naive method (forecasts for time `t` is the value observed at `t-1`).
    Useful for computing MASE forecasting error.

    As per arxiv.org/abs/2203.10716, we resort to a constant forecast based on the last-seen measurement across the entire horizon.
    By following the original measure, the naive forecaster would have the advantage of knowing the actual values whereas the predictor would not.

    Note: method assumes predictions are all for the same group combination. For a dataframe that contains multiple
     series, use `get_grouped_naive_resiudals`.

    :param target_data: observed time series targets
    :param m: season length. the naive forecasts will be the m-th previously seen value for each series

    :return: (list of naive residuals, average residual value)
    """  # noqa
    # @TODO: support categorical series as well
    residuals = np.abs(target_data.values[1:] - target_data.values[0]).flatten()
    scale_factor = np.average(residuals)
    return residuals.tolist(), scale_factor