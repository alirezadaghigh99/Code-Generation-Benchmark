def _create_ranking_fn(self, backend: BackendType) -> Callable[[List[TTensor], List[TTensor]], float]:
        """
        Creates ranking function.

        :return: The ranking function.
        """
        if self._evaluator.is_metric_mode():
            ranking_fn = operator.sub
            metric_name = "ORIGINAL"
        else:
            ranking_fn = create_normalized_mse_func(backend)
            metric_name = "NMSE"
        nncf_logger.info(f"{metric_name} metric is used to rank quantizers")

        return ranking_fn

