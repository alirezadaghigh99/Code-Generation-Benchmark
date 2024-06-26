    def from_dict(obj: Dict):
        """
        Creates a TimeseriesSettings object from python dictionary specifications.

        :param: obj: A python dictionary with the necessary representation for time-series. The only mandatory columns are ``order_by`` and ``window``.

        :returns: A populated ``TimeseriesSettings`` object.
        """ # noqa
        if len(obj) > 0:
            for mandatory_setting, etype in zip(["order_by", "window"], [str, int]):
                if mandatory_setting not in obj:
                    err = f"Missing mandatory timeseries setting: {mandatory_setting}"
                    log.error(err)
                    raise Exception(err)
                if obj[mandatory_setting] and not isinstance(obj[mandatory_setting], etype):
                    err = f"Wrong type for mandatory timeseries setting '{mandatory_setting}': found '{type(obj[mandatory_setting])}', expected '{etype}'"  # noqa
                    log.error(err)
                    raise Exception(err)

            timeseries_settings = TimeseriesSettings(
                is_timeseries=True,
                order_by=obj["order_by"],
                window=obj["window"],
                use_previous_target=obj.get("use_previous_target", True),
                historical_columns=[],
                horizon=obj.get("horizon", 1),
                allow_incomplete_history=obj.get('allow_incomplete_history', True),
                eval_incomplete=obj.get('eval_incomplete', False),
                interval_periods=obj.get('interval_periods', tuple(tuple()))
            )
            for setting in obj:
                timeseries_settings.__setattr__(setting, obj[setting])

        else:
            timeseries_settings = TimeseriesSettings(is_timeseries=False)

        return timeseries_settings