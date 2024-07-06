def get_snapshots_from_folder(train_folder: Path) -> List[str]:
    """
    Returns an ordered list of existing snapshot names in the train folder, sorted by
    increasing training iterations.

    Raises:
        FileNotFoundError: if no snapshot_names are found in the train_folder.
    """
    snapshot_names = [
        file.stem for file in train_folder.iterdir() if "index" in file.name
    ]

    if len(snapshot_names) == 0:
        raise FileNotFoundError(
            f"No snapshots were found in {train_folder}! Please ensure the network has "
            f"been trained and verify the iteration, shuffle and trainFraction are "
            f"correct."
        )

    # sort in ascending order of iteration number
    return sorted(snapshot_names, key=lambda name: int(name.split("-")[1]))

def write_config(configname, cfg):
    """
    Write structured config file.
    """
    with open(configname, "w") as cf:
        cfg_file, ruamelFile = create_config_template(
            cfg.get("multianimalproject", False)
        )
        for key in cfg.keys():
            cfg_file[key] = cfg[key]

        # Adding default value for variable skeleton and skeleton_color for backward compatibility.
        if not "skeleton" in cfg.keys():
            cfg_file["skeleton"] = []
            cfg_file["skeleton_color"] = "black"
        ruamelFile.dump(cfg_file, cf)

