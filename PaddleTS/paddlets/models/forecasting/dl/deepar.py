class DeepARModel(PaddleBaseModelImpl):
    """
    DeepAR model.

    Args:
        in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e., the number of time steps output by the model.
        rnn_type(str): The type of the specific paddle RNN module ("GRU" or "LSTM").
        hidden_size(int): The number of features in the hidden state `h` of the RNN module.
        num_layers_recurrent(int): The number of recurrent layers.
        dropout(float): The fraction of neurons that are dropped in all-but-last RNN layers.
        skip_chunk_len(int): Optional, the number of time steps between in_chunk and out_chunk for a single sample. The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample. By default it will NOT skip any time steps.
        sampling_stride(int, optional): sampling intervals between two adjacent samples.
        likelihood_model(Likelihood): The distribution likelihood to be used for probability forecasting.
        num_samples(int): The sampling number for validation and prediction phase, it is used for computation of quantiles loss and the point forecasting result.
        loss_fn(Callable[..., paddle.Tensor]): The loss fucntion of probability forecasting respect to likelihood model.
        regression_mode(str): The regression mode of prediction, `mean` and `sampling` are optional.
        output_mode(str): The mode of model output, `quantiles` and `predictions` are optional.
        optimizer_fn(Callable, Optional): optimizer algorithm.
        optimizer_params(Dict, Optional): optimizer parameters.
        eval_metrics(List[str], Optional): evaluation metrics of model.
        callbacks(List[Callback], Optional): customized callback functions.
        batch_size(int, Optional): number of samples per batch.
        max_epochs(int, Optional): max epochs during training.
        verbose(int, Optional): verbosity mode.
        patience(int, Optional): number of epochs with no improvement after which learning rate wil be reduced.
        seed(int, Optional): global random seed.
    """

    def __init__(
            self,
            in_chunk_len: int,
            out_chunk_len: int,
            rnn_type_or_module: str="LSTM",
            fcn_out_config: List[int]=None,
            hidden_size: int=128,
            num_layers_recurrent: int=1,
            dropout: float=0.0,
            skip_chunk_len: int=0,
            sampling_stride: int=1,
            likelihood_model: Likelihood=GaussianLikelihood(),
            num_samples: int=101,
            loss_fn: Callable[..., paddle.Tensor]=GaussianLikelihood().loss,
            regression_mode: str="mean",
            output_mode: str="quantiles",
            optimizer_fn: Callable[..., Optimizer]=paddle.optimizer.Adam,
            optimizer_params: Dict[str, Any]=dict(learning_rate=1e-4),
            eval_quantiles: List[float]=[0.1, 0.5, 0.9],
            eval_metrics: List[Metric]=[QuantileLoss()],
            callbacks: List[Callback]=[],
            batch_size: int=128,
            max_epochs: int=10,
            verbose: int=1,
            patience: int=4,
            seed: int=0):
        self._rnn_type_or_module = rnn_type_or_module
        self._hidden_size = hidden_size
        self._num_layers_recurrent = num_layers_recurrent
        self._dropout = dropout
        self._likelihood_model = likelihood_model
        self._num_samples = num_samples
        self._output_mode = output_mode
        self._regression_mode = regression_mode
        self._eval_quantiles = eval_quantiles
        self._q_points = [float(x) for x in list(range(OUTPUT_QUANTILE_NUM))]

        #check parameters validation
        raise_if_not(
            self._rnn_type_or_module in {"LSTM", "GRU"},
            "A valid RNN type should be specified, currently LSTM and GRU are supported."
        )

        super(DeepARModel, self).__init__(
            in_chunk_len=in_chunk_len,
            out_chunk_len=out_chunk_len,
            skip_chunk_len=skip_chunk_len,
            sampling_stride=sampling_stride,
            loss_fn=likelihood_model.loss,
            optimizer_fn=optimizer_fn,
            optimizer_params=optimizer_params,
            eval_metrics=[QuantileLoss(eval_quantiles)],
            callbacks=callbacks,
            batch_size=batch_size,
            max_epochs=max_epochs,
            verbose=verbose,
            patience=patience,
            seed=seed, )

    def _check_params(self):
        """Parameter validity verification.

        Check logic:

            batch_size: batch_size must be > 0.
            max_epochs: max_epochs must be > 0.
            verbose: verbose must be > 0.
            patience: patience must be >= 0.
        """
        raise_if(self._batch_size <= 0,
                 f"batch_size must be > 0, got {self._batch_size}.")
        raise_if(self._max_epochs <= 0,
                 f"max_epochs must be > 0, got {self._max_epochs}.")
        raise_if(self._verbose <= 0,
                 f"verbose must be > 0, got {self._verbose}.")
        raise_if(self._patience < 0,
                 f"patience must be >= 0, got {self._patience}.")
        raise_if(self._output_mode not in {"quantiles", "predictions"}, \
                 f"output mode must be one of {{`quantiles`, `predictions`}}, got `{self._output_mode}`.")
        raise_if(self._regression_mode not in {"mean", "sampling"}, \
                 f"regression mode must be one of {{`mean`, `sampling`}}, got `{self._regression_mode}`.")

        # If user does not specify an evaluation metric, a metric is provided by default.
        # Currently, only support quantile_loss
        for metric in self._eval_metrics:
            if metric != "quantile_loss" and not isinstance(metric,
                                                            QuantileLoss):
                logger.warning(
                    f"Evaluation metric is transformed to [QuantileLoss()], got {self._eval_metrics}."
                )
                self._eval_metrics = [QuantileLoss()]
                break

    def _check_tsdataset(self, tsdataset: TSDataset):
        """
        Rewrite _check_tsdataset to fit the specific model.
        For DeepAR, all data variables are expected to be float32.
        """
        for column, dtype in tsdataset.dtypes.items():
            raise_if_not(
                np.issubdtype(dtype, np.floating),
                f"deepar variables' dtype only supports [float16, float32, float64], " \
                f"but received {column}: {dtype}."
            )
        super(DeepARModel, self)._check_tsdataset(tsdataset)

    def _update_fit_params(
            self,
            train_tsdataset: List[TSDataset],
            valid_tsdataset: Optional[List[TSDataset]]=None) -> Dict[str, Any]:
        """
        Infer parameters by TSdataset automatically.

        Args:
            train_tsdataset(List[TSDataset]): list of train dataset
            valid_tsdataset(List[TSDataset], optional): list of validation dataset

        Returns:
            Dict[str, Any]: model parameters
        """
        fit_params = {
            "target_dim": train_tsdataset[0].get_target().data.shape[1],
            "known_cov_dim": 0,
            "observed_cov_dim": 0
        }
        if train_tsdataset[0].get_known_cov() is not None:
            fit_params["known_cov_dim"] = train_tsdataset[0].get_known_cov(
            ).data.shape[1]
        return fit_params

    def _init_network(self) -> paddle.nn.Layer:
        """
        Init network.

        Returns:
            paddle.nn.Layer
        """
        return _DeepAR(
            self._in_chunk_len, self._out_chunk_len,
            self._fit_params["target_dim"], self._fit_params["known_cov_dim"],
            self._rnn_type_or_module, self._hidden_size,
            self._num_layers_recurrent, self._dropout, self._likelihood_model,
            self._num_samples, self._regression_mode, self._output_mode)

    def _prepare_X_y(self, X: Dict[str, paddle.Tensor]) -> Tuple[Dict[
            str, paddle.Tensor], paddle.Tensor]:
        """Split the packet into X, y.

        Args:
            X(Dict[str, paddle.Tensor]): Dict of feature/target tensor.

        Returns:
            X(Dict[str, paddle.Tensor]): Dict of feature tensor.
            y(paddle.Tensor): Target tensor.
        """
        y = X.get("future_target", None)
        return X, y

    def _predict(
            self,
            dataloader: paddle.io.DataLoader, ) -> np.ndarray:
        """
        Predict function core logic.

        Args:
            dataloader(paddle.io.DataLoader): Data to be predicted.

        Returns:
            np.ndarray.
        """
        self._network.predicting = True
        return super(DeepARModel, self)._predict(dataloader)

