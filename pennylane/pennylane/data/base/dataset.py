def open(
        cls,
        filepath: Union[str, Path],
        mode: Literal["w", "w-", "a", "r", "copy"] = "r",
    ) -> "Dataset":
        """Open existing dataset or create a new one at ``filepath``.

        Args:
            filepath: Path to dataset file
            mode: File handling mode. Possible values are "w-" (create, fail if file
                exists), "w" (create, overwrite existing), "a" (append existing,
                create if doesn't exist), "r" (read existing, must exist), and "copy",
                which loads the dataset into memory and detaches it from the underlying
                file. Default is "r".
        Returns:
            Dataset object from file
        """
        filepath = Path(filepath).expanduser()

        if mode == "copy":
            with h5py.File(filepath, "r") as f_to_copy:
                f = hdf5.create_group()
                hdf5.copy_all(f_to_copy, f)
        else:
            f = h5py.File(filepath, mode)

        return cls(f)

