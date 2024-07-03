def compute_col_stats(
    ser: Series,
    stype: torch_frame.stype,
    sep: str | None = None,
    time_format: str | None = None,
) -> dict[StatType, Any]:
    if stype == torch_frame.numerical:
        ser = ser.mask(ser.isin([np.inf, -np.inf]), np.nan)
        if not ptypes.is_numeric_dtype(ser):
            raise TypeError("Numerical series contains invalid entries. "
                            "Please make sure your numerical series "
                            "contains only numerical values or nans.")
    if ser.isnull().all():
        # NOTE: We may just error out here if eveything is NaN
        stats = {
            stat_type: _default_values[stat_type]
            for stat_type in StatType.stats_for_stype(stype)
        }
    else:
        if stype == torch_frame.timestamp:
            ser = pd.to_datetime(ser, format=time_format)
            ser = ser.sort_values()
        stats = {
            stat_type: stat_type.compute(ser.dropna(), sep)
            for stat_type in StatType.stats_for_stype(stype)
        }

    return stats