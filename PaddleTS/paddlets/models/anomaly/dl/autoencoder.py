class AutoEncoder(AnomalyBaseModel):
    """Auto encoder network for anomaly detection.

    Args:
        in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        sampling_stride(int): Sampling intervals between two adjacent samples.
        loss_fn(Callable[..., paddle.Tensor]): Loss function.
        optimizer_fn(Callable[..., Optimizer]): Optimizer algorithm.
        threshold_fn(Callable[..., float]|None): The method to get anomaly threshold.
        q(float): The parameter used to calculate the quantile which range is [0, 100].
        threshold(float|None): The threshold to judge anomaly.
        anomaly_score_fn(Callable[..., List[float]]|None): The method to get anomaly score.
        pred_adjust(bool): Whether to adjust the pred label according to the real label.
        pred_adjust_fn(Callable[..., np.ndarray]|None): The method to adjust pred label.
        optimizer_params(Dict[str, Any]): Optimizer parameters.
        eval_metrics(List[str]): Evaluation metrics of model.
        callbacks(List[Callback]): Customized callback functions.
        batch_size(int): Number of samples per batch.
        max_epochs(int): Max epochs during training.
        verbose(int): Verbosity mode.
        patience(int): Number of epochs to wait for improvement before terminating.
        seed(int|None): Global random seed.

        ed_type(str): The type of encoder and decoder.
        activation(Callable[..., paddle.Tensor]): The activation function for the hidden layers.
        last_layer_activation(Callable[..., paddle.Tensor]): The activation function for the last layer.
        hidden_config(List[int]|None): The ith element represents the number of neurons in the ith hidden layer.
        kernel_size(int): Kernel size for Conv1D.
        dropout_rate(float): Dropout regularization parameter.
        use_bn(bool): Whether to use batch normalization.
        embedding_size(int): The size of each embedding vector.
        pooling: Whether to use average pooling to aggregate embeddings, if False, concat each embedding.

    Attributes:
        _in_chunk_len(int): The size of the loopback window, i.e. the number of time steps feed to the model.
        _sampling_stride(int): Sampling intervals between two adjacent samples.
        _loss_fn(Callable[..., paddle.Tensor]): Loss function.
        _optimizer_fn(Callable[..., Optimizer]): Optimizer algorithm.
        _threshold_fn(Callable[..., float]|None)): The method to get anomaly threshold.
        _q(float): The parameter used to calculate the quantile which range is [0, 100].
        _threshold(float|None): The threshold to judge anomaly.
        _anomaly_score_fn(Callable[..., List[float]]|None): The method to get anomaly score.
        _pred_adjust(bool): Whether to adjust the pred label according to the real label.
        _pred_adjust_fn(Callable[..., np.ndarray]|None): The method to adjust pred label.
        _optimizer_params(Dict[str, Any]): Optimizer parameters.
        _eval_metrics(List[str]): Evaluation metrics of model.
        _callbacks(List[Callback]): Customized callback functions.
        _batch_size(int): Number of samples per batch.
        _max_epochs(int): Max epochs during training.
        _verbose(int): Verbosity mode.
        _patience(int): Number of epochs to wait for improvement before terminating.
        _seed(int|None): Global random seed.
        _stop_training(bool): Training status.
        _ed_type(str): The type of encoder and decoder.
        _activation(Callable[..., paddle.Tensor]): The activation function for the hidden layers.
        _last_layer_activation(Callable[..., paddle.Tensor]): The activation function for the last layer.
        _hidden_config(List[int]|None): The ith element represents the number of neurons in the ith hidden layer.
        _kernel_size(int): Kernel size for Conv1D.
        _dropout_rate(float): Dropout regularization parameter.
        _use_bn(bool): Whether to use batch normalization.
        _embedding_size(int): The size of each embedding vector.
        _pooling(bool): Whether to use average pooling to aggregate embeddings, if False, concat each embedding.
    """

    def __init__(
            self,
            in_chunk_len: int,
            sampling_stride: int=1,
            loss_fn: Callable[..., paddle.Tensor]=F.mse_loss,
            optimizer_fn: Callable[..., Optimizer]=paddle.optimizer.Adam,
            threshold_fn: Callable[..., float]=U.percentile,
            q: float=100,
            threshold: Optional[float]=None,
            threshold_coeff: float=1.0,
            anomaly_score_fn: Callable[..., List[float]]=None,
            pred_adjust: bool=False,
            pred_adjust_fn: Callable[..., np.ndarray]=U.result_adjust,
            optimizer_params: Dict[str, Any]=dict(learning_rate=1e-3),
            eval_metrics: List[str]=[],
            callbacks: List[Callback]=[],
            batch_size: int=32,
            max_epochs: int=100,
            verbose: int=1,
            patience: int=10,
            seed: Optional[int]=None,
            ed_type: str='MLP',
            activation: Callable[..., paddle.Tensor]=paddle.nn.ReLU,
            last_layer_activation: Callable[...,
                                            paddle.Tensor]=paddle.nn.Identity,
            use_bn: bool=False,
            hidden_config: List[int]=None,
            kernel_size: int=3,
            dropout_rate: float=0.2,
            embedding_size: int=16,
            pooling: bool=False, ):
        self._hidden_config = (hidden_config if hidden_config else [32, 16])
        self._use_bn = use_bn
        self._kernel_size = kernel_size
        self._ed_type = ed_type
        self._activation = activation
        self._last_layer_activation = last_layer_activation
        self._dropout_rate = dropout_rate
        self._embedding_size = embedding_size
        self._pooling = pooling
        self._q = q

        super(AutoEncoder, self).__init__(
            in_chunk_len=in_chunk_len,
            sampling_stride=sampling_stride,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            threshold=threshold,
            threshold_coeff=threshold_coeff,
            threshold_fn=threshold_fn,
            anomaly_score_fn=anomaly_score_fn,
            pred_adjust=pred_adjust,
            pred_adjust_fn=pred_adjust_fn,
            optimizer_params=optimizer_params,
            eval_metrics=eval_metrics,
            callbacks=callbacks,
            batch_size=batch_size,
            max_epochs=max_epochs,
            verbose=verbose,
            patience=patience,
            seed=seed, )

    def _update_fit_params(
            self,
            train_tsdataset: TSDataset,
            valid_tsdataset: Optional[TSDataset]=None) -> Dict[str, Any]:
        """Infer parameters by TSdataset automatically.

        Args:
            train_tsdataset(TSDataset): train dataset.
            valid_tsdataset(TSDataset|None): validation dataset.

        Returns:
            Dict[str, Any]: model parameters.
        """
        train_df = train_tsdataset.to_dataframe()
        observed_cat_cols = collections.OrderedDict()
        observed_num_cols = []
        observed_train_tsdataset = train_tsdataset.get_observed_cov()
        observed_dtypes = dict(observed_train_tsdataset.dtypes)
        for col in observed_train_tsdataset.columns:
            if np.issubdtype(observed_dtypes[col], np.integer):
                observed_cat_cols[col] = len(train_df[col].unique())
            else:
                observed_num_cols.append(col)

        fit_params = {
            "observed_cat_cols": observed_cat_cols,
            "observed_num_dim": len(observed_num_cols),
            "observed_cat_dim": len(observed_cat_cols),
        }
        return fit_params

    def _init_network(self) -> paddle.nn.Layer:
        """Setup the network.

        Returns:
            paddle.nn.Layer.
        """
        return _AEBlock(self._in_chunk_len, self._ed_type, self._fit_params,
                        self._hidden_config, self._activation,
                        self._last_layer_activation, self._kernel_size,
                        self._dropout_rate, self._use_bn, self._embedding_size,
                        self._pooling)

    def _get_threshold(self, anomaly_score: np.ndarray) -> float:
        """Get the threshold value to judge anomaly.
        
        Args:
            anomaly_score(np.ndarray): 
            
        Returns:
            float: Thresold value.
        """
        return self._threshold_fn(anomaly_score, self._q)

