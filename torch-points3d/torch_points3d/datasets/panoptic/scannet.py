class ScannetDataset(BaseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        # Update to OmegaConf 2.0
        use_instance_labels: bool = dataset_opt.get('use_instance_labels')
        donotcare_class_ids: [] = list(dataset_opt.get('donotcare_class_ids', []))
        max_num_point: int = dataset_opt.get('max_num_point', None)
        is_test: bool = dataset_opt.get('is_test', False)

        self.train_dataset = ScannetPanoptic(
            self._data_path,
            split="train",
            pre_transform=self.pre_transform,
            transform=self.train_transform,
            version=dataset_opt.version,
            use_instance_labels=use_instance_labels,
            use_instance_bboxes=False,
            donotcare_class_ids=donotcare_class_ids,
            max_num_point=max_num_point,
            is_test=is_test,
        )

        self.val_dataset = ScannetPanoptic(
            self._data_path,
            split="val",
            transform=self.val_transform,
            pre_transform=self.pre_transform,
            version=dataset_opt.version,
            use_instance_labels=use_instance_labels,
            use_instance_bboxes=False,
            donotcare_class_ids=donotcare_class_ids,
            max_num_point=max_num_point,
            is_test=is_test,
        )

    @property  # type: ignore
    @save_used_properties
    def stuff_classes(self):
        """ Returns a list of classes that are not instances
        """
        return self.train_dataset.stuff_classes

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
        Returns:
            [BaseTracker] -- tracker
        """
        return PanopticTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)

