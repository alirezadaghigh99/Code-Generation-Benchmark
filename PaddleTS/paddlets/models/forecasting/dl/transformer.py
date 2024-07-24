class TransformerModel(PaddleBaseModelImpl):
    """Transformer\[1\] is a state-of-the-art deep learning model introduced in 2017. 
    It is an encoder-decoder architecture whose core feature is the `multi-head attention` mechanism, 
    which is able to draw intra-dependencies within the input vector and within the output vector (`self-attention`)
    as well as inter-dependencies between input and output vectors (`encoder-decoder attention`).

    \[1\] Vaswani A, et al. "Attention Is All You Need", `<https://arxiv.org/abs/1706.03762>`_

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

        d_model(int): The expected feature size for the input/output of the transformer's encoder/decoder.
        nhead(int): The number of heads in the multi-head attention mechanism.
        num_encoder_layers(int): The number of encoder layers in the encoder.
        num_decoder_layers(int): The number of decoder layers in the decoder.
        dim_feedforward(int): The dimension of the feedforward network model.
        activation(str): The activation function of encoder/decoder intermediate layer, ["relu", "gelu"] is optional.
        dropout_rate(float): Fraction of neurons affected by Dropout.
        custom_encoder(paddle.nn.Layer|None): A custom user-provided encoder module for the transformer.
        custom_decoder(paddle.nn.Layer|None): A custom user-provided decoder module for the transformer.

    Attributes:
        _in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        _out_chunk_len(int): The size of the forecasting horizon, i.e. the number of time steps output by the model.
        _skip_chunk_len(int): Optional, the number of time steps between in_chunk and out_chunk for a single sample.
            The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample.
            By default it will NOT skip any time steps.
        _sampling_stride(int): Sampling intervals between two adjacent samples.
        _loss_fn(Callable[..., paddle.Tensor]|None): Loss function.
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

        _d_model(int): The expected feature size for the input/output of the transformer's encoder/decoder.
        _nhead(int): The number of heads in the multi-head attention mechanism.
        _num_encoder_layers(int): The number of encoder layers in the encoder.
        _num_decoder_layers(int): The number of decoder layers in the decoder.
        _dim_feedforward(int): The dimension of the feedforward network model.
        _activation(str): The activation function of encoder/decoder intermediate layer. ["relu", "gelu"] is optional.
        _dropout_rate(float): Fraction of neurons affected by Dropout.
        _custom_encoder(paddle.nn.Layer|None): A custom user-provided encoder module for the transformer.
        _custom_decoder(paddle.nn.Layer|None): A custom user-provided decoder module for the transformer.
    """

    def __init__(
            self,
            in_chunk_len: int,
            out_chunk_len: int,
            skip_chunk_len: int=0,
            sampling_stride: int=1,
            loss_fn: Callable[..., paddle.Tensor]=F.mse_loss,
            optimizer_fn: Callable[..., Optimizer]=paddle.optimizer.Adam,
            optimizer_params: Dict[str, Any]=dict(learning_rate=1e-3),
            eval_metrics: List[str]=[],
            callbacks: List[Callback]=[],
            batch_size: int=128,
            max_epochs: int=10,
            verbose: int=1,
            patience: int=4,
            seed: Optional[int]=None,
            d_model: int=8,
            nhead: int=4,
            num_encoder_layers: int=1,
            num_decoder_layers: int=1,
            dim_feedforward: int=64,
            activation: str="relu",
            dropout_rate: float=0.1,
            custom_encoder: Optional[paddle.nn.Layer]=None,
            custom_decoder: Optional[paddle.nn.Layer]=None, ):
        self._d_model = d_model
        self._nhead = nhead
        self._num_encoder_layers = num_encoder_layers
        self._num_decoder_layers = num_decoder_layers
        self._dim_feedforward = dim_feedforward
        self._activation = activation
        self._dropout_rate = dropout_rate
        self._custom_encoder = custom_encoder
        self._custom_decoder = custom_decoder
        super(TransformerModel, self).__init__(
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
                    f"transformer's target dtype only supports [float16, float32, float64], " \
                    f"but received {column}: {dtype}."
                )
                continue
            raise_if_not(
                np.issubdtype(dtype, np.floating),
                f"transformer's cov(observed or known) dtype currently only supports [float16, float32, float64], " \
                f"but received {column}: {dtype}."
            )
        super(TransformerModel, self)._check_tsdataset(tsdataset)

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
        known_num_dim = 0
        observed_num_dim = 0
        input_dim = target_dim = train_tsdataset[0].get_target().data.shape[1]
        if train_tsdataset[0].get_observed_cov():
            observed_num_dim = train_tsdataset[0].get_observed_cov(
            ).data.shape[1]
            input_dim += observed_num_dim
        if train_tsdataset[0].get_known_cov():
            known_num_dim = train_tsdataset[0].get_known_cov().data.shape[1]
            input_dim += known_num_dim
        fit_params = {
            "target_dim": target_dim,
            "input_dim": input_dim,
            "known_num_dim": known_num_dim,
            "observed_num_dim": observed_num_dim,
        }
        return fit_params

    def _init_network(self) -> paddle.nn.Layer:
        """Setup the network.

        Returns:
            paddle.nn.Layer
        """
        return _TransformerModule(
            in_chunk_len=self._in_chunk_len,
            out_chunk_len=self._out_chunk_len,
            target_dim=self._fit_params["target_dim"],
            input_dim=self._fit_params["input_dim"],
            d_model=self._d_model,
            nhead=self._nhead,
            num_encoder_layers=self._num_encoder_layers,
            num_decoder_layers=self._num_decoder_layers,
            dim_feedforward=self._dim_feedforward,
            activation=self._activation,
            dropout_rate=self._dropout_rate,
            custom_encoder=self._custom_encoder,
            custom_decoder=self._custom_decoder)

