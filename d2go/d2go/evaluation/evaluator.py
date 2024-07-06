def inference_on_dataset(
    model: torch.nn.Module,
    data_loader: Iterable,
    evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None],
    **kwargs,
):
    """
    A drop-in replacement for d2's inference_on_dataset to run inference on datasets,
    supports customization for checkpointing
    * has_finished_process(self) -> bool: return True if `self.process()` could be skipped
    """
    if evaluator is None:
        return inference_on_dataset_d2(model, data_loader, evaluator, **kwargs)

    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)

    if not (
        hasattr(evaluator, "has_finished_process") and evaluator.has_finished_process()
    ):
        return inference_on_dataset_d2(model, data_loader, evaluator, **kwargs)

    evaluator.reset()
    results = evaluator.evaluate()
    if results is None:
        results = {}
    return results

