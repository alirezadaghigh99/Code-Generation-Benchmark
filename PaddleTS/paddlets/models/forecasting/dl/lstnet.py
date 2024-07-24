class LSTNetRegressor(PaddleBaseModelImpl):
    """LSTNet\[1\] is a time series forecasting model introduced in 2018. LSTNet uses the 
    Convolution Neural Network (CNN) and the Recurrent Neural Network (RNN) to extract short-term local 
    dependency patterns among variables and to discover long-term patterns for time series trends.

    \[1\] Lai G, et al. "Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks", `<https://arxiv.org/abs/1703.07015>`_

    Args:
        in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e. the number of time steps output by the model.
        skip_chunk_len(int): Optional, the number of time steps between in_chunk and out_chunk for a single sample.
            The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample.
            By default it will NOT skip any time steps.
        sampling_stride(int): Sampling intervals between two adjacent samples.
        loss_fn(Callable[..., paddle.Tensor]|None): Loss function.
        optimizer_fn(Callable[..., Optimizer]): Optimizer algorithm.
        optimizer_params(Dict[str, Any]): Optimizer parameters.
        eval_metrics(List[str]): Evaluation metrics of model.
        callbacks(List[Callback]): Customized callback functions.
        batch_size(int): Number of samples per batch.
        max_epochs(int): Max epochs during training.
        verbose(int): Verbosity mode.
        patience(int): Number of epochs to wait for improvement before terminating.
        seed(int|None): Global random seed.

        skip_size(int): Skip size for the skip RNN layer.
        channels(int): Number of channels for first layer Conv1D.
        kernel_size(int): Kernel size for first layer Conv1D.
        rnn_cell_type(str): Type of the RNN cell, Either GRU or LSTM.
        rnn_num_cells(int): Number of RNN cells for each layer.
        skip_rnn_cell_type(str): Type of the RNN cell for the skip layer, Either GRU or LSTM.
        skip_rnn_num_cells(int): Number of RNN cells for each layer for skip part.
        dropout_rate(float): Dropout regularization parameter.
        output_activation(str|None): The last activation to be used for output.
            Accepts either None (default no activation), sigmoid or tanh.

    Attributes:
        _in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        _out_chunk_len(int): The size of the forecasting horizon, i.e. the number of time steps output by the model.
        _skip_chunk_len(int): Optional, the number of time steps between in_chunk and out_chunk for a single sample.
            The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample.
            By default it will NOT skip any time steps.
        _sampling_stride(int): Sampling intervals between two adjacent samples.
        _loss_fn(Callable[..., paddle.Tensor]): Loss function.
        _optimizer_fn(Callable[..., Optimizer]): Optimizer algorithm.
        _optimizer_params(Dict[str, Any]): Optimizer parameters.
        _eval_metrics(List[str]): Evaluation metrics of model.
        _callbacks(List[Callback]): Customized callback functions.
        _batch_size(int): Number of samples per batch.
        _max_epochs(int): Max epochs during training.
        _verbose(int): Verbosity mode.
        _patience(int): Number of epochs to wait for improvement before terminating.
        _seed(int|None): Global random seed.
        _stop_training(bool) Training status.
        _skip_size(int): Skip size for the skip RNN layer.
        _channels(int): Number of channels for first layer Conv1D.
        _kernel_size(int): Kernel size for first layer Conv1D.
        _rnn_cell_type(str): Type of the RNN cell, Either GRU or LSTM.
        _rnn_num_cells(int): Number of RNN cells for each layer.
        _skip_rnn_cell_type(str): Type of the RNN cell for the skip layer, Either GRU or LSTM.
        _skip_rnn_num_cells(int): Number of RNN cells for each layer for skip part.
        _dropout_rate(float): Dropout regularization parameter.
        _output_activation(str|None): The last activation to be used for output.
            Accepts either None (default no activation), sigmoid or tanh.
    """

    def __init__(self,
                 in_chunk_len: int,
                 out_chunk_len: int,
                 skip_chunk_len: int=0,
                 sampling_stride: int=1,
                 loss_fn: Callable[..., paddle.Tensor]=F.mse_loss,
                 optimizer_fn: Callable[..., Optimizer]=paddle.optimizer.Adam,
                 optimizer_params: Dict[str, Any]=dict(learning_rate=1e-3),
                 eval_metrics: List[str]=[],
                 callbacks: List[Callback]=[],
                 batch_size: int=32,
                 max_epochs: int=100,
                 verbose: int=1,
                 patience: int=10,
                 seed: Optional[int]=None,
                 skip_size: int=1,
                 channels: int=1,
                 kernel_size: int=3,
                 rnn_cell_type: str="GRU",
                 rnn_num_cells: int=10,
                 skip_rnn_cell_type: str="GRU",
                 skip_rnn_num_cells: int=10,
                 dropout_rate: float=0.2,
                 output_activation: Optional[str]=None):
        self._skip_size = skip_size
        self._channels = channels
        self._kernel_size = kernel_size
        self._rnn_cell_type = rnn_cell_type
        self._rnn_num_cells = rnn_num_cells
        self._skip_rnn_cell_type = skip_rnn_cell_type
        self._skip_rnn_num_cells = skip_rnn_num_cells
        self._dropout_rate = dropout_rate
        self._output_activation = output_activation
        super(LSTNetRegressor, self).__init__(
            in_chunk_len=in_chunk_len,
            out_chunk_len=out_chunk_len,
            skip_chunk_len=skip_chunk_len,
            sampling_stride=sampling_stride,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            optimizer_params=optimizer_params,
            eval_metrics=eval_metrics,
            callbacks=callbacks,
            batch_size=batch_size,
            max_epochs=max_epochs,
            verbose=verbose,
            patience=patience,
            seed=seed, )

    def _check_tsdataset(self, tsdataset: TSDataset):
        """Ensure the robustness of input data (consistent feature order), at the same time,
            check whether the data types are compatible. If not, the processing logic is as follows:

            1> Integer: Convert to np.int64.

            2> Floating: Convert to np.float32.

            3> Missing value: Warning.

            4> Other: Illegal.

        Args:
            tsdataset(TSDataset): Data to be checked.
        """
        target_columns = tsdataset.get_target().dtypes.keys()
        for column, dtype in tsdataset.dtypes.items():
            if column in target_columns:
                raise_if_not(
                    np.issubdtype(dtype, np.floating),
                    f"lstnet's target dtype only supports [float16, float32, float64], " \
                    f"but received {column}: {dtype}."
                )
                continue
            raise_if_not(
                np.issubdtype(dtype, np.floating),
                f"lstnet's cov(observed or known) dtype currently only supports [float16, float32, float64], " \
                f"but received {column}: {dtype}."
            )
        super(LSTNetRegressor, self)._check_tsdataset(tsdataset)

    def _update_fit_params(
            self,
            train_tsdataset: List[TSDataset],
            valid_tsdataset: Optional[List[TSDataset]]=None) -> Dict[str, Any]:
        """Infer parameters by TSdataset automatically.

        Args:
            train_tsdataset(List[TSDataset]): list of train dataset.
            valid_tsdataset(List[TSDataset]|None): list of validation dataset.

        Returns:
            Dict[str, Any]: model parameters.
        """
        target_dim = train_tsdataset[0].get_target().data.shape[1]
        fit_params = {"target_dim": target_dim}
        return fit_params

    def _init_network(self) -> paddle.nn.Layer:
        """Setup the network.

        Returns:
            paddle.nn.Layer.
        """
        return _LSTNetModule(
            in_chunk_len=self._in_chunk_len,
            out_chunk_len=self._out_chunk_len,
            target_dim=self._fit_params["target_dim"],
            skip_size=self._skip_size,
            channels=self._channels,
            kernel_size=self._kernel_size,
            rnn_cell_type=self._rnn_cell_type,
            rnn_num_cells=self._rnn_num_cells,
            skip_rnn_cell_type=self._skip_rnn_cell_type,
            skip_rnn_num_cells=self._skip_rnn_num_cells,
            dropout_rate=self._dropout_rate,
            output_activation=self._output_activation)

