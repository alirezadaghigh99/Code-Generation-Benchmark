class CNNClassifier(PaddleBaseClassifier):
    """CNNClassifier.

    Args:
        loss_fn(Callable[..., paddle.Tensor]): Loss function.
        optimizer_fn(Callable[..., Optimizer]): Optimizer algorithm.
        optimizer_params(Dict[str, Any]): Optimizer parameters.
        eval_metrics(List[str]): Evaluation metrics of model.
        callbacks(List[Callback]): Customized callback functions.
        batch_size(int): Number of samples per batch.
        max_epochs(int): Max epochs during training.
        verbose(int): Verbosity mode.
        patience(int): Number of epochs to wait for improvement before terminating.
        seed(int|None): Global random seed.

        activation(Callable[..., paddle.Tensor]): The activation function for the hidden layers.
        last_activation(Callable[..., paddle.Tensor]) : The activation function for the last hidden layers.
        hidden_config(List[int]|None): The ith element represents the number of neurons in the ith hidden layer.
        kernel_size(int): Kernel size for Conv1D.
        dropout_rate(float): Dropout regularization parameter.
        use_bn(bool): Whether to use batch normalization.

    Attributes:
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
        _stop_training(bool): Training status.
        _activation(Callable[..., paddle.Tensor]): The activation function for the hidden layers.
        _last_activation(Callable[..., paddle.Tensor]) : The activation function for the last hidden layers.
        _hidden_config(List[int]|None): The ith element represents the number of neurons in the ith hidden layer.
        _kernel_size(int): Kernel size for Conv1D.
        _dropout_rate(float): Dropout regularization parameter.
        _use_bn(bool): Whether to use batch normalization.
    """

    def __init__(
            self,
            loss_fn: Callable[..., paddle.Tensor]=F.cross_entropy,
            optimizer_fn: Callable[..., Optimizer]=paddle.optimizer.Adam,
            optimizer_params: Dict[str, Any]=dict(learning_rate=1e-3),
            eval_metrics: List[str]=[],
            callbacks: List[Callback]=[],
            batch_size: int=32,
            max_epochs: int=100,
            verbose: int=1,
            patience: int=10,
            seed: Optional[int]=None,
            activation: Callable[..., paddle.Tensor]=paddle.nn.Sigmoid,
            last_activation: Callable[..., paddle.Tensor]=paddle.nn.Softmax,
            use_bn: bool=False,
            hidden_config: List[int]=[6, 12],
            kernel_size: int=7,
            avg_pool_size=3,
            dropout_rate: float=0.2,
            use_drop: bool=False, ):
        self._hidden_config = hidden_config
        self._use_bn = use_bn
        self._kernel_size = kernel_size
        self._avg_pool_size = avg_pool_size
        self._activation = activation
        self._last_activation = last_activation
        self._dropout_rate = dropout_rate
        self._use_drop = use_drop
        self._dropout_rate = dropout_rate

        super(CNNClassifier, self).__init__(
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

    def _update_fit_params(self,
                           train_tsdatasets: List[TSDataset],
                           train_labels: np.ndarray,
                           valid_tsdatasets: List[TSDataset],
                           valid_labels: np.ndarray) -> Dict[str, Any]:
        """Infer parameters by TSdataset automatically.

        Args:
            train_tsdataset(TSDataset): Train set.
            train_labels:(np.ndarray) : The train data class labels
            valid_tsdataset(TSDataset|None): Eval set, used for early stopping.
            valid_labels:(np.ndarray) : The valid data class labels
        Returns:
            Dict[str, Any]: model parameters.
        """
        fit_params = {
            "feature_dim": train_tsdatasets[0].get_target().data.shape[1],
            "input_lens": train_tsdatasets[0].get_target().data.shape[0]
        }
        return fit_params

    def _init_network(self) -> paddle.nn.Layer:
        """Setup the network.

        Returns:
            paddle.nn.Layer.
        """
        return _CNNBlock(
            self._fit_params["feature_dim"],
            self._fit_params["input_lens"],
            self._n_classes,
            self._hidden_config,
            self._activation,
            self._last_activation,
            self._kernel_size,
            self._avg_pool_size,
            self._use_bn,
            self._use_drop,
            self._dropout_rate, )

