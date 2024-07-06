def from_dataset(cls, dataset: TimeSeriesDataSet, **kwargs) -> LightningModule:
        """
        Create model from dataset, i.e. save dataset parameters in model

        This function should be called as ``super().from_dataset()`` in a derived models that implement it

        Args:
            dataset (TimeSeriesDataSet): timeseries dataset

        Returns:
            BaseModel: Model that can be trained
        """
        if "output_transformer" not in kwargs:
            kwargs["output_transformer"] = dataset.target_normalizer
        if "dataset_parameters" not in kwargs:
            kwargs["dataset_parameters"] = dataset.get_parameters()
        net = cls(**kwargs)
        if dataset.multi_target:
            assert isinstance(
                net.loss, MultiLoss
            ), f"multiple targets require loss to be MultiLoss but found {net.loss}"
        else:
            assert not isinstance(net.loss, MultiLoss), "MultiLoss not compatible with single target"

        return net

