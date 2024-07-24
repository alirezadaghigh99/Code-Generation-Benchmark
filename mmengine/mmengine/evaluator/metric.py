class DumpResults(BaseMetric):
    """Dump model predictions to a pickle file for offline evaluation.

    Args:
        out_file_path (str): Path of the dumped file. Must end with '.pkl'
            or '.pickle'.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        collect_dir: (str, optional): Synchronize directory for collecting data
            from different ranks. This argument should only be configured when
            ``collect_device`` is 'cpu'. Defaults to None.
            `New in version 0.7.3.`
    """

    def __init__(self,
                 out_file_path: str,
                 collect_device: str = 'cpu',
                 collect_dir: Optional[str] = None) -> None:
        super().__init__(
            collect_device=collect_device, collect_dir=collect_dir)
        if not out_file_path.endswith(('.pkl', '.pickle')):
            raise ValueError('The output file must be a pkl file.')
        self.out_file_path = out_file_path

    def process(self, data_batch: Any, predictions: Sequence[dict]) -> None:
        """transfer tensors in predictions to CPU."""
        self.results.extend(_to_cpu(predictions))

    def compute_metrics(self, results: list) -> dict:
        """dump the prediction results to a pickle file."""
        dump(results, self.out_file_path)
        print_log(
            f'Results has been saved to {self.out_file_path}.',
            logger='current')
        return {}

