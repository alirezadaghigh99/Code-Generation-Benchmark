class NBEATSModel(PaddleBaseModelImpl):
    """
    Implementation of NBeats model.

    Args:
        in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e., the number of time steps output by the model.
        generic_architecture(bool, Optional): Boolean value indicating whether the generic architecture of N-BEATS is used. \
                    If not, the interpretable architecture outlined in the paper (consisting of one trend and one seasonality stack \
                    with appropriate waveform generator functions).
        num_stacks(int, Optional): The number of stacks that make up the whole model. Only used if `generic_architecture` is set to `True`.
        num_blocks(Union[int, List[int]], Optional): The number of blocks making up each stack. \
                    If a list is passed, it must have a length equal to `num_stacks` and every entry in that list corresponds to the corresponding stack.\
                    If an integer is passed, every stack will have the same number of blocks.
        num_layers(int, Optional): The number of fully connected layers preceding the final forking layers in each block of every stack. \
                    Only used if `generic_architecture` is set to `True`.
        layer_widths(Union[int, List[int]], Optional): Determines the number of neurons that make up each fully connected layer in each block of every stack. If a list is passed, it must have a length equal to `num_stacks` and every entry in that list corresponds to the layer width of the corresponding stack. If an integer is passed, every stack will have blocks with FC layers of the same width.
        expansion_coefficient_dim(int, Optional): The dimensionality of the waveform generator parameters, also known as expansion coefficients. Only used if `generic_architecture` is set to `True`.
        trend_polynomial_degree(int, Optional): The degree of the polynomial used as waveform generator in trend stacks. Only used if `generic_architecture` is set to `False`.
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
                 generic_architecture: bool=True,
                 num_stacks: int=2,
                 num_blocks: Union[int, List[int]]=3,
                 num_layers: int=4,
                 layer_widths: Union[int, List[int]]=128,
                 expansion_coefficient_dim: int=128,
                 trend_polynomial_degree: int=4,
                 skip_chunk_len: int=0,
                 sampling_stride: int=1,
                 loss_fn: Callable[..., paddle.Tensor]=F.mse_loss,
                 optimizer_fn: Callable[..., Optimizer]=paddle.optimizer.Adam,
                 optimizer_params: Dict[str, Any]=dict(learning_rate=1e-4),
                 use_revin: bool=False,
                 revin_params: Dict[str, Any]=dict(
                     eps=1e-5, affine=True),
                 eval_metrics: List[str]=[],
                 callbacks: List[Callback]=[],
                 batch_size: int=32,
                 max_epochs: int=10,
                 verbose: int=1,
                 patience: int=10,
                 seed: int=0):
        self._generic_architecture = generic_architecture
        self._num_stacks = num_stacks
        self._num_blocks = num_blocks
        self._num_layers = num_layers
        self._layer_widths = layer_widths
        self._expansion_coefficient_dim = expansion_coefficient_dim
        self._trend_polynomial_degree = trend_polynomial_degree
        self._use_revin = use_revin
        self._revin_params = revin_params
        # If not using general architecture, for interpretable purpose, number of stacks is forced to be 2, for trend and seasonality implementation
        if not self._generic_architecture:
            self._num_stacks = 2
        if isinstance(self._num_blocks, int):
            self._num_blocks = [self._num_blocks] * self._num_stacks
        if isinstance(self._layer_widths, int):
            self._layer_widths = [self._layer_widths] * self._num_stacks

        super(NBEATSModel, self).__init__(
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
        Check validation of parameters.
        """
        raise_if(
            isinstance(self._num_blocks, list) and
            len(self._num_blocks) != self._num_stacks,
            "Stack number should be equal to the length of the List: num_blocks."
        )
        raise_if(
            isinstance(self._layer_widths, list) and
            len(self._layer_widths) != self._num_stacks,
            "Stack number should be equal to the length of the List: layer_widths."
        )
        super(NBEATSModel, self)._check_params()

    def _check_tsdataset(self, tsdataset: TSDataset):
        """ 
        Rewrite _check_tsdataset to fit the specific model.
        For NBeats, all data variables are expected to be float32.
        """
        for column, dtype in tsdataset.dtypes.items():
            raise_if_not(
                np.issubdtype(dtype, np.floating),
                f"nbeats variables' dtype only supports [float16, float32, float64], " \
                f"but received {column}: {dtype}."
            )
        super(NBEATSModel, self)._check_tsdataset(tsdataset)

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
        fit_params = {
            "target_dim": train_tsdataset[0].get_target().data.shape[1],
            "known_cov_dim": 0,
            "observed_cov_dim": 0
        }
        if train_tsdataset[0].get_known_cov() is not None:
            fit_params["known_cov_dim"] = train_tsdataset[0].get_known_cov(
            ).data.shape[1]
        if train_tsdataset[0].get_observed_cov() is not None:
            fit_params["observed_cov_dim"] = train_tsdataset[
                0].get_observed_cov().data.shape[1]
        return fit_params

    @revin_norm
    def _init_network(self) -> paddle.nn.Layer:
        """
        Init network.

        Returns:
            paddle.nn.Layer
        """
        return _NBEATSModule(
            self._in_chunk_len, self._out_chunk_len,
            self._fit_params["target_dim"], self._fit_params["known_cov_dim"],
            self._fit_params["observed_cov_dim"], self._generic_architecture,
            self._num_stacks, self._num_blocks, self._num_layers,
            self._layer_widths, self._expansion_coefficient_dim,
            self._trend_polynomial_degree)

