def split(
        self,
        by: str | list[int] | list[list[int]] | dict[str, list[int]] | None = None,
        property: str | None = None,
        split_ids: list[int] | list[list[int]] | dict[str, list[int]] | None = None,
    ) -> dict[str, BaseConcatDataset]:
        """Split the dataset based on information listed in its description
        DataFrame or based on indices.

        Parameters
        ----------
        by : str | list | dict
            If ``by`` is a string, splitting is performed based on the
            description DataFrame column with this name.
            If ``by`` is a (list of) list of integers, the position in the first
            list corresponds to the split id and the integers to the
            datapoints of that split.
            If a dict then each key will be used in the returned
            splits dict and each value should be a list of int.
        property : str
            Some property which is listed in info DataFrame.
        split_ids : list | dict
            List of indices to be combined in a subset.
            It can be a list of int or a list of list of int.

        Returns
        -------
        splits : dict
            A dictionary with the name of the split (a string) as key and the
            dataset as value.
        """
        args_not_none = [by is not None, property is not None, split_ids is not None]
        if sum(args_not_none) != 1:
            raise ValueError("Splitting requires exactly one argument.")

        if property is not None or split_ids is not None:
            warnings.warn(
                "Keyword arguments `property` and `split_ids` "
                "are deprecated and will be removed in the future. "
                "Use `by` instead.",
                DeprecationWarning,
            )
            by = property if property is not None else split_ids
        if isinstance(by, str):
            split_ids = {
                k: list(v) for k, v in self.description.groupby(by).groups.items()
            }
        elif isinstance(by, dict):
            split_ids = by
        else:
            # assume list(int)
            if not isinstance(by[0], list):
                by = [by]
            # assume list(list(int))
            split_ids = {split_i: split for split_i, split in enumerate(by)}

        return {
            str(split_name): BaseConcatDataset(
                [self.datasets[ds_ind] for ds_ind in ds_inds],
                target_transform=self.target_transform,
            )
            for split_name, ds_inds in split_ids.items()
        }

