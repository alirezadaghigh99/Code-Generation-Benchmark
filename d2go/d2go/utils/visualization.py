class DataLoaderVisWrapper:
    """
    Wrap the data loader to visualize its output via TensorBoardX at given frequency.
    """

    def __init__(
        self,
        cfg,
        tbx_writer,
        data_loader,
        visualizer: Optional[Type[VisualizerWrapper]] = None,
    ):
        self.tbx_writer = tbx_writer
        self.data_loader = data_loader
        self._visualizer = visualizer(cfg) if visualizer else VisualizerWrapper(cfg)

        self.log_frequency = cfg.TENSORBOARD.TRAIN_LOADER_VIS_WRITE_PERIOD
        self.log_limit = cfg.TENSORBOARD.TRAIN_LOADER_VIS_MAX_IMAGES
        self.batch_log_limit = cfg.TENSORBOARD.TRAIN_LOADER_VIS_MAX_BATCH_IMAGES
        assert self.log_frequency >= 0
        assert self.log_limit >= 0
        assert self.batch_log_limit >= 0
        self._remaining = self.log_limit

    def __iter__(self):
        for data in self.data_loader:
            self._maybe_write_vis(data)
            yield data

    def _maybe_write_vis(self, data):
        try:
            storage = get_event_storage()
        except AssertionError:
            # wrapped data loader might be used outside EventStorage, don't visualize
            # anything
            return

        if (
            self.log_frequency == 0
            or not storage.iter % self.log_frequency == 0
            or self._remaining <= 0
        ):
            return

        length = min(len(data), min(self.batch_log_limit, self._remaining))
        data = data[:length]
        self._remaining -= length

        for i, per_image in enumerate(data):
            vis_image = self._visualizer.visualize_train_input(per_image)
            tag = [f"train_loader_batch_{storage.iter}"]
            if "dataset_name" in per_image:
                tag += [per_image["dataset_name"]]
            if "file_name" in per_image:
                tag += [f"img_{i}", per_image["file_name"]]

            if isinstance(vis_image, dict):
                for k in vis_image:
                    self.tbx_writer._writer.add_image(
                        tag="/".join(tag + [k]),
                        img_tensor=vis_image[k],
                        global_step=storage.iter,
                        dataformats="HWC",
                    )
            else:
                self.tbx_writer._writer.add_image(
                    tag="/".join(tag),
                    img_tensor=vis_image,
                    global_step=storage.iter,
                    dataformats="HWC",
                )

