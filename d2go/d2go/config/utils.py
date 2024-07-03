def get_cfg_diff_table(cfg, original_cfg):
    """
    Print the different of two config dicts side-by-side in a table
    """

    all_old_keys = list(flatten_config_dict(original_cfg, reorder=True).keys())
    all_new_keys = list(flatten_config_dict(cfg, reorder=True).keys())

    diff_table = []
    if all_old_keys != all_new_keys:
        logger = logging.getLogger(__name__)
        mismatched_old_keys = set(all_old_keys) - set(all_new_keys)
        mismatched_new_keys = set(all_new_keys) - set(all_old_keys)
        logger.warning(
            "Config key mismatched.\n"
            f"Mismatched old keys: {mismatched_old_keys}\n"
            f"Mismatched new keys: {mismatched_new_keys}"
        )
        for old_key in mismatched_old_keys:
            old_value = get_from_flattened_config_dict(original_cfg, old_key)
            diff_table.append([old_key, old_value, "Key not exists"])

        for new_key in mismatched_new_keys:
            new_value = get_from_flattened_config_dict(cfg, new_key)
            diff_table.append([new_key, "Key not exists", new_value])

        # filter out mis-matched keys
        all_old_keys = [x for x in all_old_keys if x not in mismatched_old_keys]
        all_new_keys = [x for x in all_new_keys if x not in mismatched_new_keys]

    for full_key in all_new_keys:
        old_value = get_from_flattened_config_dict(original_cfg, full_key)
        new_value = get_from_flattened_config_dict(cfg, full_key)
        if old_value != new_value:
            diff_table.append([full_key, old_value, new_value])

    from tabulate import tabulate

    table = tabulate(
        diff_table,
        tablefmt="pipe",
        headers=["config key", "old value", "new value"],
    )
    return table