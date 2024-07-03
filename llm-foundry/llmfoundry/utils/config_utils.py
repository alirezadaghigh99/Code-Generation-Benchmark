def _parse_source_dataset(cfg: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """Parse a run config for dataset information.

    Given a config dictionary, parse through it to determine what the datasource
    should be categorized as. Possible data sources are Delta Tables, UC Volumes,
    HuggingFace paths, remote storage, or local storage.

    Args:
        cfg (DictConfig): A config dictionary of a run

    Returns:
        List[Tuple[str, str, str]]: A list of tuples formatted as (data type, path, split)
    """
    data_paths = []

    # Handle train loader if it exists
    train_dataset: Dict = cfg.get('train_loader', {}).get('dataset', {})
    train_split = train_dataset.get('split', None)
    train_source_path = cfg.get('source_dataset_train', None)
    _process_data_source(
        train_source_path,
        train_dataset,
        train_split,
        'train',
        data_paths,
    )

    # Handle eval_loader which might be a list or a single dictionary
    eval_data_loaders = cfg.get('eval_loader', {})
    if not isinstance(eval_data_loaders, list):
        eval_data_loaders = [
            eval_data_loaders,
        ]  # Normalize to list if it's a single dictionary

    for eval_data_loader in eval_data_loaders:
        assert isinstance(eval_data_loader, dict)  # pyright type check
        eval_dataset: Dict = eval_data_loader.get('dataset', {})
        eval_split = eval_dataset.get('split', None)
        eval_source_path = cfg.get('source_dataset_eval', None)
        _process_data_source(
            eval_source_path,
            eval_dataset,
            eval_split,
            'eval',
            data_paths,
        )

    return data_paths