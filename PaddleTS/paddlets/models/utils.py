def to_tsdataset(scenario: str="forecasting") -> Callable[..., Callable[
        ..., TSDataset]]:
    """A decorator, used for converting ndarray to tsdataset 
    (compatible with both DL and ML, compatible with both forecasting and anomaly).

    Args:
        scenario(str): The task type. ["forecasting", "anomaly_label", "anomaly_score"] is optional.

    Returns:
        Callable[..., Callable[..., TSDataset]]: Wrapped core function.
    """

    def decorate(func) -> Callable[..., TSDataset]:
        @functools.wraps(func)
        def wrapper(obj: BaseModel, tsdataset: TSDataset,
                    **kwargs) -> TSDataset:
            """Core processing logic.

            Args:
                obj(BaseModel): BaseModel instance.
                tsdataset(TSDataset): tsdataset.

            Returns:
                TSDataset: tsdataset.
            """
            raise_if_not(
                scenario in ("forecasting", "anomaly_label", "anomaly_score"),
                f"{scenario} not supported, ['forecasting', 'anomaly_label', 'anomaly_score'] is optional."
            )

            results = func(obj, tsdataset, **kwargs)
            if scenario == "anomaly_label" or scenario == "anomaly_score":
                # Generate target cols
                target_cols = tsdataset.get_target()
                if target_cols is None:
                    target_cols = [scenario]
                else:
                    target_cols = target_cols.data.columns
                    if scenario == "anomaly_score":
                        target_cols = target_cols + '_score'
                # Generate target index freq
                target_index = tsdataset.get_observed_cov().data.index
                if isinstance(target_index, pd.RangeIndex):
                    freq = target_index.step
                else:
                    freq = target_index.freqstr
                results_size = results.size
                raise_if(
                    results_size == 0,
                    f"There is something wrong, anomaly predict size is 0, you'd better check the tsdataset or the predict logic."
                )
                target_index = target_index[-results_size:]
                anomaly_target = pd.DataFrame(
                    results, index=target_index, columns=target_cols)
                return TSDataset.load_from_dataframe(anomaly_target, freq=freq)

            past_target_index = tsdataset.get_target().data.index
            if isinstance(past_target_index, pd.RangeIndex):
                freq = past_target_index.step
                future_target_index = pd.RangeIndex(
                    past_target_index[-1] + (1 + obj._skip_chunk_len) * freq,
                    past_target_index[-1] +
                    (1 + obj._skip_chunk_len + obj._out_chunk_len) * freq,
                    step=freq)
            else:
                freq = past_target_index.freqstr
                future_target_index = pd.date_range(
                    past_target_index[-1] +
                    (1 + obj._skip_chunk_len) * past_target_index.freq,
                    periods=obj._out_chunk_len,
                    freq=freq)
            target_cols = tsdataset.get_target().data.columns
            # for probability forecasting and quantile output
            if hasattr(obj, "_output_mode"
                       ) and obj._output_mode == QUANTILE_OUTPUT_MODE:
                target_cols = [
                    x + "@" + "quantile" + str(y)
                    for x in target_cols for y in obj._q_points
                ]
            future_target = pd.DataFrame(
                np.reshape(
                    results, newshape=[obj._out_chunk_len, -1]),
                index=future_target_index,
                columns=target_cols)
            return TSDataset.load_from_dataframe(future_target, freq=freq)

        return wrapper

    return decorate

