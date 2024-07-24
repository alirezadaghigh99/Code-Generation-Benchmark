class NHiTSModel(PaddleBaseModelImpl):
    """
    Implementation of NHiTS model

    Args:
        in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e., the number of time steps output by the model.
        num_stacks: The number of stacks that make up the whole model.
        num_blocks: The number of blocks making up every stack.
        num_layers: The number of fully connected layers preceding the final forking layers in each block of every stack.
        layer_widths: Determines the number of neurons that make up each fully connected layer in each block of every stack. If a list is passed, it must have a length equal to `num_stacks` and every entry in that list corresponds to the layer width of the corresponding stack. If an integer is passed, every stack will have blocks with FC layers of the same width.
        pooling_kernel_size(Tuple[Tuple[int]], option): The kernel size for the initial pooling layer.
        n_freq_downsample(Tuple[Tuple[int]], option): The factor by which to downsample time at the output (before interpolating).
        batch_norm(bool): Whether to use batch normalization.
        dropout(float): Dropout probability.
        activation(str): The activation function of encoder/decoder intermediate layer.
        MaxPool1d(bool): Whether to use MaxPool1d pooling, False uses AvgPool1d.
        skip_chunk_len(int, Optional): Optional, the number of time steps between in_chunk and out_chunk for a single sample. The skip chunk is neither used as a feature (i.e. X) nor a label (i.e. Y) for a single sample. By default it will NOT skip any time steps.
        sampling_stride(int, optional): sampling intervals between two adjacent samples.
        loss_fn(Callable, Optional): loss function.
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

    def __init__(self,
                 in_chunk_len: int,
                 out_chunk_len: int,
                 num_stacks: int=3,
                 num_blocks: int=3,
                 num_layers: int=2,
                 layer_widths: Union[int, List[int]]=512,
                 pooling_kernel_sizes: Optional[Tuple[Tuple[int]]]=None,
                 n_freq_downsample: Optional[Tuple[Tuple[int]]]=None,
                 batch_norm: bool=False,
                 dropout: float=0.1,
                 activation: str="ReLU",
                 MaxPool1d: bool=True,
                 skip_chunk_len: int=0,
                 sampling_stride: int=1,
                 loss_fn: Callable[..., paddle.Tensor]=F.mse_loss,
                 optimizer_fn: Callable[..., Optimizer]=paddle.optimizer.Adam,
                 optimizer_params: Dict[str, Any]=dict(learning_rate=1e-4),
                 eval_metrics: List[str]=[],
                 callbacks: List[Callback]=[],
                 batch_size: int=256,
                 max_epochs: int=10,
                 verbose: int=1,
                 patience: int=4,
                 seed: int=0):
        self._num_stacks = num_stacks
        self._num_blocks = num_blocks
        self._num_layers = num_layers
        self._layer_widths = layer_widths
        self._pooling_kernel_sizes = pooling_kernel_sizes
        self._n_freq_downsample = n_freq_downsample

        self._activation = activation
        self._MaxPool1d = MaxPool1d
        self._dropout = dropout
        self._batch_norm = batch_norm

        if isinstance(self._layer_widths, int):
            self._layer_widths = [self._layer_widths] * self._num_stacks

        super(NHiTSModel, self).__init__(
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

    def _check_params(self):
        """
        check validation of parameters
        """
        raise_if(
            isinstance(self._layer_widths, list) and
            len(self._layer_widths) != self._num_stacks,
            "Stack number should be equal to the length of the List: layer_widths."
        )
        super(NHiTSModel, self)._check_params()

    def _check_tsdataset(self, tsdataset: TSDataset):
        """-
        Rewrite _check_tsdataset to fit the specific model.
        For NHiTS, all data variables are expected to be float32.
        """
        for column, dtype in tsdataset.dtypes.items():
            raise_if_not(
                np.issubdtype(dtype, np.floating),
                f"nhits variables' dtype only supports [float16, float32, float64], " \
                f"but received {column}: {dtype}."
            )
        super(NHiTSModel, self)._check_tsdataset(tsdataset)

    def _update_fit_params(
            self,
            train_tsdataset: List[TSDataset],
            valid_tsdataset: Optional[List[TSDataset]]=None) -> Dict[str, Any]:
        """
        Infer parameters by TSDataset automatically.

        Args:
            train_tsdataseet(List[TSDataset]): list of train dataset
            valid_tsdataset(List[TSDataset], optional): list of validation dataset
        Returns:
            Dict[str, Any]: model parameters
        """
        train_ts0 = train_tsdataset[0]
        fit_params = {
            "target_dim": train_ts0.get_target().data.shape[1],
            "known_cov_dim": 0,
            "observed_cov_dim": 0
        }
        if train_ts0.get_known_cov() is not None:
            fit_params["known_cov_dim"] = train_ts0.get_known_cov().data.shape[
                1]
        if train_ts0.get_observed_cov() is not None:
            fit_params["observed_cov_dim"] = train_ts0.get_observed_cov(
            ).data.shape[1]
        return fit_params

    def _init_network(self) -> paddle.nn.Layer:
        """
        init network

        Returns:
            paddle.nn.Layer
        """

        return _NHiTSModule(
            self._in_chunk_len,
            self._out_chunk_len,
            self._fit_params["target_dim"],
            self._fit_params["known_cov_dim"],
            self._fit_params["observed_cov_dim"],
            self._num_stacks,
            self._num_blocks,
            self._num_layers,
            self._layer_widths,
            self._pooling_kernel_sizes,
            self._n_freq_downsample,
            self._batch_norm,
            self._dropout,
            self._activation,
            self._MaxPool1d, )

