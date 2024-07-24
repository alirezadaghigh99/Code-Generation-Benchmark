class DatasetBenchmark(BaseBenchmark):
    """The dataset benchmark class. It will be statistical inference FPS, FPS
    pre transform and CPU memory information.

    Args:
        cfg (mmengine.Config): config.
        dataset_type (str): benchmark data type, only supports ``train``,
            ``val`` and ``test``.
        max_iter (int): maximum iterations of benchmark. Defaults to 2000.
        log_interval (int): interval of logging. Defaults to 50.
        num_warmup (int): Number of Warmup. Defaults to 5.
        logger (MMLogger, optional): Formatted logger used to record messages.
    """

    def __init__(self,
                 cfg: Config,
                 dataset_type: str,
                 max_iter: int = 2000,
                 log_interval: int = 50,
                 num_warmup: int = 5,
                 logger: Optional[MMLogger] = None):
        super().__init__(max_iter, log_interval, num_warmup, logger)
        assert dataset_type in ['train', 'val', 'test'], \
            'dataset_type only supports train,' \
            f' val and test, but got {dataset_type}'
        assert get_world_size(
        ) == 1, 'Dataset benchmark does not allow distributed multi-GPU'
        self.cfg = copy.deepcopy(cfg)

        if dataset_type == 'train':
            dataloader_cfg = copy.deepcopy(cfg.train_dataloader)
        elif dataset_type == 'test':
            dataloader_cfg = copy.deepcopy(cfg.test_dataloader)
        else:
            dataloader_cfg = copy.deepcopy(cfg.val_dataloader)

        dataset_cfg = dataloader_cfg.pop('dataset')
        dataset = DATASETS.build(dataset_cfg)
        if hasattr(dataset, 'full_init'):
            dataset.full_init()
        self.dataset = dataset

    def run_once(self) -> dict:
        """Executes the benchmark once."""
        pure_inf_time = 0
        fps = 0

        total_index = list(range(len(self.dataset)))
        np.random.shuffle(total_index)

        start_time = time.perf_counter()
        for i, idx in enumerate(total_index):
            if (i + 1) % self.log_interval == 0:
                print_log('==================================', self.logger)

            get_data_info_start_time = time.perf_counter()
            data_info = self.dataset.get_data_info(idx)
            get_data_info_elapsed = time.perf_counter(
            ) - get_data_info_start_time

            if (i + 1) % self.log_interval == 0:
                print_log(f'get_data_info - {get_data_info_elapsed * 1000} ms',
                          self.logger)

            for t in self.dataset.pipeline.transforms:
                transform_start_time = time.perf_counter()
                data_info = t(data_info)
                transform_elapsed = time.perf_counter() - transform_start_time

                if (i + 1) % self.log_interval == 0:
                    print_log(
                        f'{t.__class__.__name__} - '
                        f'{transform_elapsed * 1000} ms', self.logger)

                if data_info is None:
                    break

            elapsed = time.perf_counter() - start_time

            if i >= self.num_warmup:
                pure_inf_time += elapsed
                if (i + 1) % self.log_interval == 0:
                    fps = (i + 1 - self.num_warmup) / pure_inf_time

                    print_log(
                        f'Done img [{i + 1:<3}/{self.max_iter}], '
                        f'fps: {fps:.1f} img/s, '
                        f'times per img: {1000 / fps:.1f} ms/img', self.logger)

            if (i + 1) == self.max_iter:
                fps = (i + 1 - self.num_warmup) / pure_inf_time
                break

            start_time = time.perf_counter()

        return {'fps': fps}

    def average_multiple_runs(self, results: List[dict]) -> dict:
        """Average the results of multiple runs."""
        print_log('============== Done ==================', self.logger)

        fps_list_ = [round(result['fps'], 1) for result in results]
        avg_fps_ = sum(fps_list_) / len(fps_list_)
        outputs = {'avg_fps': avg_fps_, 'fps_list': fps_list_}

        if len(fps_list_) > 1:
            times_pre_image_list_ = [
                round(1000 / result['fps'], 1) for result in results
            ]
            avg_times_pre_image_ = sum(times_pre_image_list_) / len(
                times_pre_image_list_)

            print_log(
                f'Overall fps: {fps_list_}[{avg_fps_:.1f}] img/s, '
                'times per img: '
                f'{times_pre_image_list_}[{avg_times_pre_image_:.1f}] '
                'ms/img', self.logger)
        else:
            print_log(
                f'Overall fps: {fps_list_[0]:.1f} img/s, '
                f'times per img: {1000 / fps_list_[0]:.1f} ms/img',
                self.logger)

        return outputs

