class ParametricUMAP(UMAP):
    def __init__(
        self,
        batch_size=None,
        dims=None,
        encoder=None,
        decoder=None,
        parametric_reconstruction=False,
        parametric_reconstruction_loss_fcn=None,
        parametric_reconstruction_loss_weight=1.0,
        autoencoder_loss=False,
        reconstruction_validation=None,
        global_correlation_loss_weight=0,
        keras_fit_kwargs={},
        **kwargs
    ):
        """
        Parametric UMAP subclassing UMAP-learn, based on keras/tensorflow.
        There is also a non-parametric implementation contained within to compare
        with the base non-parametric implementation.

        Parameters
        ----------
        batch_size : int, optional
            size of batch used for batch training, by default None
        dims :  tuple, optional
            dimensionality of data, if not flat (e.g. (32x32x3 images for ConvNet), by default None
        encoder : keras.Sequential, optional
            The encoder Keras network
        decoder : keras.Sequential, optional
            the decoder Keras network
        parametric_reconstruction : bool, optional
            Whether the decoder is parametric or non-parametric, by default False
        parametric_reconstruction_loss_fcn : bool, optional
            What loss function to use for parametric reconstruction,
            by default keras.losses.BinaryCrossentropy
        parametric_reconstruction_loss_weight : float, optional
            How to weight the parametric reconstruction loss relative to umap loss, by default 1.0
        autoencoder_loss : bool, optional
            [description], by default False
        reconstruction_validation : array, optional
            validation X data for reconstruction loss, by default None
        global_correlation_loss_weight : float, optional
            Whether to additionally train on correlation of global pairwise relationships (>0), by default 0
        keras_fit_kwargs : dict, optional
            additional arguments for model.fit (like callbacks), by default {}
        """
        super().__init__(**kwargs)

        # add to network
        self.dims = dims  # if this is an image, we should reshape for network
        self.encoder = encoder  # neural network used for embedding
        self.decoder = decoder  # neural network used for decoding
        self.parametric_reconstruction = parametric_reconstruction
        self.parametric_reconstruction_loss_weight = (
            parametric_reconstruction_loss_weight
        )
        self.parametric_reconstruction_loss_fcn = parametric_reconstruction_loss_fcn
        self.autoencoder_loss = autoencoder_loss
        self.batch_size = batch_size
        self.loss_report_frequency = 10
        self.global_correlation_loss_weight = global_correlation_loss_weight

        self.reconstruction_validation = (
            reconstruction_validation  # holdout data for reconstruction acc
        )
        self.keras_fit_kwargs = keras_fit_kwargs  # arguments for model.fit
        self.parametric_model = None

        # How many epochs to train for
        # (different than n_epochs which is specific to each sample)
        self.n_training_epochs = 1

        # Set optimizer.
        # Adam is better for parametric_embedding. Use gradient clipping by value.
        self.optimizer = keras.optimizers.Adam(1e-3, clipvalue=4.0)

        if self.encoder is not None:
            if encoder.outputs[0].shape[-1] != self.n_components:
                raise ValueError(
                    (
                        "Dimensionality of embedder network output ({}) does"
                        "not match n_components ({})".format(
                            encoder.outputs[0].shape[-1], self.n_components
                        )
                    )
                )

    def fit(self, X, y=None, precomputed_distances=None):
        if self.metric == "precomputed":
            if precomputed_distances is None:
                raise ValueError(
                    "Precomputed distances must be supplied if metric \
                    is precomputed."
                )
            # prepare X for training the network
            self._X = X
            # geneate the graph on precomputed distances
            return super().fit(precomputed_distances, y)
        else:
            return super().fit(X, y)

    def fit_transform(self, X, y=None, precomputed_distances=None):

        if self.metric == "precomputed":
            if precomputed_distances is None:
                raise ValueError(
                    "Precomputed distances must be supplied if metric \
                    is precomputed."
                )
            # prepare X for training the network
            self._X = X
            # generate the graph on precomputed distances
            return super().fit_transform(precomputed_distances, y)
        else:
            return super().fit_transform(X, y)

    def transform(self, X):
        """Transform X into the existing embedded space and return that
        transformed output.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            New data to be transformed.
        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedding of the new data in low-dimensional space.
        """
        return self.encoder.predict(
            np.asanyarray(X), batch_size=self.batch_size, verbose=self.verbose
        )

    def inverse_transform(self, X):
        """ Transform X in the existing embedded space back into the input
        data space and return that transformed output.
        Parameters
        ----------
        X : array, shape (n_samples, n_components)
            New points to be inverse transformed.
        Returns
        -------
        X_new : array, shape (n_samples, n_features)
            Generated data points new data in data space.
        """
        if self.parametric_reconstruction:
            return self.decoder.predict(
                np.asanyarray(X), batch_size=self.batch_size, verbose=self.verbose
            )
        else:
            return super().inverse_transform(X)

    def _define_model(self):
        """Define the model in keras"""
        prlw = self.parametric_reconstruction_loss_weight
        self.parametric_model = UMAPModel(
            self._a,
            self._b,
            negative_sample_rate=self.negative_sample_rate,
            encoder=self.encoder,
            decoder=self.decoder,
            parametric_reconstruction_loss_fn=self.parametric_reconstruction_loss_fcn,
            parametric_reconstruction=self.parametric_reconstruction,
            parametric_reconstruction_loss_weight=prlw,
            global_correlation_loss_weight=self.global_correlation_loss_weight,
            autoencoder_loss=self.autoencoder_loss,
        )

    def _fit_embed_data(self, X, n_epochs, init, random_state):

        if self.metric == "precomputed":
            X = self._X

        # get dimensionality of dataset
        if self.dims is None:
            self.dims = [np.shape(X)[-1]]
        else:
            # reshape data for network
            if len(self.dims) > 1:
                X = np.reshape(X, [len(X)] + list(self.dims))

        if self.parametric_reconstruction and (np.max(X) > 1.0 or np.min(X) < 0.0):
            warn(
                "Data should be scaled to the range 0-1 for cross-entropy reconstruction loss."
            )

        # get dataset of edges
        (
            edge_dataset,
            self.batch_size,
            n_edges,
            head,
            tail,
            self.edge_weight,
        ) = construct_edge_dataset(
            X,
            self.graph_,
            self.n_epochs,
            self.batch_size,
            self.parametric_reconstruction,
            self.global_correlation_loss_weight,
        )
        self.head = ops.array(ops.expand_dims(head.astype(np.int64), 0))
        self.tail = ops.array(ops.expand_dims(tail.astype(np.int64), 0))

        init_embedding = None

        # create encoder and decoder model
        n_data = len(X)
        self.encoder, self.decoder = prepare_networks(
            self.encoder,
            self.decoder,
            self.n_components,
            self.dims,
            n_data,
            self.parametric_reconstruction,
            init_embedding,
        )

        # create the model
        self._define_model()

        # report every loss_report_frequency subdivision of an epochs
        steps_per_epoch = int(
            n_edges / self.batch_size / self.loss_report_frequency
        )

        # Validation dataset for reconstruction
        if (
            self.parametric_reconstruction
            and self.reconstruction_validation is not None
        ):

            # reshape data for network
            if len(self.dims) > 1:
                self.reconstruction_validation = np.reshape(
                    self.reconstruction_validation,
                    [len(self.reconstruction_validation)] + list(self.dims),
                )

            validation_data = (
                (
                    self.reconstruction_validation,
                    ops.zeros_like(self.reconstruction_validation),
                ),
                {"reconstruction": self.reconstruction_validation},
            )
        else:
            validation_data = None

        # create embedding
        history = self.parametric_model.fit(
            edge_dataset,
            epochs=self.loss_report_frequency * self.n_training_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_data,
            **self.keras_fit_kwargs
        )
        # save loss history dictionary
        self._history = history.history

        # get the final embedding
        embedding = self.encoder.predict(X, verbose=self.verbose)

        return embedding, {}

    def __getstate__(self):
        # this function supports pickling, making sure that objects can be pickled
        return dict(
            (k, v)
            for (k, v) in self.__dict__.items()
            if should_pickle(k, v) and k not in ("optimizer", "encoder", "decoder", "parametric_model")
        )

    def save(self, save_location, verbose=True):

        # save encoder
        if self.encoder is not None:
            encoder_output = os.path.join(save_location, "encoder.keras")
            self.encoder.save(encoder_output)
            if verbose:
                print("Keras encoder model saved to {}".format(encoder_output))

        # save decoder
        if self.decoder is not None:
            decoder_output = os.path.join(save_location, "decoder.keras")
            self.decoder.save(decoder_output)
            if verbose:
                print("Keras decoder model saved to {}".format(decoder_output))

        # save parametric_model
        if self.parametric_model is not None:
            parametric_model_output = os.path.join(save_location, "parametric_model.keras")
            self.parametric_model.save(parametric_model_output)
            if verbose:
                print("Keras full model saved to {}".format(parametric_model_output))

        # # save model.pkl (ignoring unpickleable warnings)
        with catch_warnings():
            filterwarnings("ignore")
            model_output = os.path.join(save_location, "model.pkl")
            with open(model_output, "wb") as output:
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
            if verbose:
                print("Pickle of ParametricUMAP model saved to {}".format(model_output))

