class RNNBlockRegressor(PaddleBaseModelImpl):
    """
    Implementation of RNN Block model.

    Args:
        in_chunk_len(int): The size of the loopback window, i.e., the number of time steps feed to the model.
        out_chunk_len(int): The size of the forecasting horizon, i.e., the number of time steps output by the model.
        rnn_type_or_module(str, Optional): The type of the specific paddle RNN module ("SimpleRNN", "GRU" or "LSTM").
        fcn_out_config(List[int], Optional): A list containing the dimensions of the hidden layers of the fully connected NN.
        hidden_size(int, Optional): The number of features in the hidden state `h` of the RNN module.
        embedding_size(int, Optional): The size of each embedding vector.
        num_layers_recurrent(int, Optional): The number of recurrent layers.
        dropout(float, Optional): The fraction of neurons that are dropped in all-but-last RNN layers.
        pooling(bool, Optional): Whether to use average pooling to aggregate embeddings, if False, concat each embedding.
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
                 rnn_type_or_module: str="SimpleRNN",
                 fcn_out_config: List[int]=None,
                 hidden_size: int=128,
                 embedding_size: int=128,
                 num_layers_recurrent: int=1,
                 dropout: float=0.0,
                 pooling: bool=True,
                 skip_chunk_len: int=0,
                 sampling_stride: int=1,
                 loss_fn: Callable[..., paddle.Tensor]=F.mse_loss,
                 optimizer_fn: Callable[..., Optimizer]=paddle.optimizer.Adam,
                 optimizer_params: Dict[str, Any]=dict(learning_rate=1e-4),
                 eval_metrics: List[str]=[],
                 callbacks: List[Callback]=[],
                 batch_size: int=128,
                 max_epochs: int=10,
                 verbose: int=1,
                 patience: int=4,
                 seed: int=0):
        self._rnn_type_or_module = rnn_type_or_module
        self._fcn_out_config = fcn_out_config
        self._hidden_size = hidden_size
        self._embedding_size = embedding_size
        self._num_layers_recurrent = num_layers_recurrent
        self._pooling = pooling
        self._dropout = dropout

        #check parameters validation
        raise_if_not(
            self._rnn_type_or_module in {"SimpleRNN", "LSTM", "GRU"},
            "A valid RNN type should be specified, currently SimpleRNN, LSTM, and GRU are supported."
        )

        super(RNNBlockRegressor, self).__init__(
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
        """
        Rewrite _check_tsdataset to fit the specific model.
        For RNN, all data variables are expected to be float32.
        """
        target_columns = tsdataset.get_target().dtypes.keys()
        for column, dtype in tsdataset.dtypes.items():
            if column in target_columns:
                raise_if_not(
                    np.issubdtype(dtype, np.floating),
                    f"rnn's target dtype only supports [float16, float32, float64]," \
                    f"but received {column}: {dtype}."
                )
            else:
                raise_if_not(
                    np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.integer),
                    f"rnn's covariates' dtype only support float and integer," \
                    f"but received {column}: {dtype}."
                )
        super(RNNBlockRegressor, self)._check_tsdataset(tsdataset)

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
        df_list = []
        static_info = []
        # for meta info of all tsdatasets
        all_ts = train_tsdataset + valid_tsdataset if valid_tsdataset else train_tsdataset
        for ts in all_ts:
            static = ts.get_static_cov()
            df = ts.to_dataframe()
            if static:
                for col, val in static.items():
                    df[col] = val
            df_list.append(df)
        df_all = pd.concat(df_list)

        train_ts0 = train_tsdataset[0]
        train_ts0.sort_columns()
        target_dim = train_ts0.get_target().data.shape[1]
        # stat categorical variables' dict size
        # known info
        known_cat_size = collections.OrderedDict()
        known_ts = train_ts0.get_known_cov()
        known_num_cols = []
        if known_ts:
            known_dtypes = dict(known_ts.dtypes)
            for col in known_ts.columns:
                if np.issubdtype(known_dtypes[col], np.integer):
                    known_cat_size[col] = len(df_all[col].unique())
                else:
                    known_num_cols.append(col)
        #observed info
        observed_cat_size = collections.OrderedDict()
        observed_ts = train_ts0.get_observed_cov()
        observed_num_cols = []
        if observed_ts:
            observed_dtypes = dict(observed_ts.dtypes)
            for col in observed_ts.columns:
                if np.issubdtype(observed_dtypes[col], np.integer):
                    observed_cat_size[col] = len(df_all[col].unique())
                else:
                    observed_num_cols.append(col)
        # static info
        static_cat_size = collections.OrderedDict()
        static_dic = train_ts0.get_static_cov()
        static_num_cols = []
        if static_dic:
            for col, val in static_dic.items():
                if np.issubdtype(type(val), np.integer) or isinstance(val,
                                                                      int):
                    static_cat_size[col] = len(df_all[col].unique())
                else:
                    static_num_cols.append(col)

        fit_params = {
            "target_dim": target_dim,
            "known_num_dim": len(known_num_cols),
            "known_cat_dim": len(known_cat_size),
            "observed_num_dim": len(observed_num_cols),
            "observed_cat_dim": len(observed_cat_size),
            "static_num_dim": len(static_num_cols),
            "static_cat_dim": len(static_cat_size),
            "known_cat_size": known_cat_size,
            "observed_cat_size": observed_cat_size,
            "static_cat_size": static_cat_size,
        }
        return fit_params

    def _init_network(self) -> paddle.nn.Layer:
        """
        Init network.

        Returns:
            paddle.nn.Layer
        """
        return _RNNBlock(self._in_chunk_len, self._out_chunk_len,
                         self._fit_params, self._rnn_type_or_module,
                         self._hidden_size, self._embedding_size,
                         self._num_layers_recurrent, self._fcn_out_config,
                         self._dropout, self._pooling)

