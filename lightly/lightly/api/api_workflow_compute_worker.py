def selection_config_from_dict(cfg: Dict[str, Any]) -> SelectionConfigV4:
    """Recursively converts selection config from dict to a SelectionConfigV4 instance."""
    strategies = []
    for entry in cfg.get("strategies", []):
        new_entry = copy.deepcopy(entry)
        new_entry["input"] = SelectionConfigV4EntryInput(**entry["input"])
        new_entry["strategy"] = SelectionConfigV4EntryStrategy(**entry["strategy"])
        strategies.append(SelectionConfigV4Entry(**new_entry))
    new_cfg = copy.deepcopy(cfg)
    new_cfg["strategies"] = strategies
    return SelectionConfigV4(**new_cfg)

