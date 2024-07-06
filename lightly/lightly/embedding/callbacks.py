def create_summary_callback(
    summary_callback_config: DictConfig, trainer_config: DictConfig
) -> ModelSummary:
    """Creates a summary callback."""
    # TODO: Drop support for the "weights_summary" argument.
    weights_summary = trainer_config.get("weights_summary", None)
    if weights_summary not in [None, "None"]:
        return _create_summary_callback_deprecated(weights_summary)
    else:
        return _create_summary_callback(summary_callback_config["max_depth"])

