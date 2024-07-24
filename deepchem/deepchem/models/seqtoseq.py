class AspuruGuzikAutoEncoder(SeqToSeq):
    """
    This is an implementation of Automatic Chemical Design Using a Continuous Representation of Molecules
    http://pubs.acs.org/doi/full/10.1021/acscentsci.7b00572

    Abstract
    --------
    We report a method to convert discrete representations of molecules to and
    from a multidimensional continuous representation. This model allows us to
    generate new molecules for efficient exploration and optimization through
    open-ended spaces of chemical compounds. A deep neural network was trained on
    hundreds of thousands of existing chemical structures to construct three
    coupled functions: an encoder, a decoder, and a predictor. The encoder
    converts the discrete representation of a molecule into a real-valued
    continuous vector, and the decoder converts these continuous vectors back to
    discrete molecular representations. The predictor estimates chemical
    properties from the latent continuous vector representation of the molecule.
    Continuous representations of molecules allow us to automatically generate
    novel chemical structures by performing simple operations in the latent space,
    such as decoding random vectors, perturbing known chemical structures, or
    interpolating between molecules. Continuous representations also allow the use
    of powerful gradient-based optimization to efficiently guide the search for
    optimized functional compounds. We demonstrate our method in the domain of
    drug-like molecules and also in a set of molecules with fewer that nine heavy
    atoms.

    Notes
    -------
    This is currently an imperfect reproduction of the paper.  One difference is
    that teacher forcing in the decoder is not implemented.  The paper also
    discusses co-learning molecular properties at the same time as training the
    encoder/decoder.  This is not done here.  The hyperparameters chosen are from
    ZINC dataset.

    This network also currently suffers from exploding gradients.  Care has to be taken when training.

    NOTE(LESWING): Will need to play around with annealing schedule to not have exploding gradients
    TODO(LESWING): Teacher Forcing
    TODO(LESWING): Sigmoid variational loss annealing schedule
    The output GRU layer had one
    additional input, corresponding to the character sampled from the softmax output of the
    previous time step and was trained using teacher forcing. 48 This increased the accuracy
    of generated SMILES strings, which resulted in higher fractions of valid SMILES strings
    for latent points outside the training data, but also made training more difficult, since the
    decoder showed a tendency to ignore the (variational) encoding and rely solely on the input
    sequence. The variational loss was annealed according to sigmoid schedule after 29 epochs,
    running for a total 120 epochs

    I also added a BatchNorm before the mean and std embedding layers.  This has empiracally
    made training more stable, and is discussed in Ladder Variational Autoencoders.
    https://arxiv.org/pdf/1602.02282.pdf
    Maybe if Teacher Forcing and Sigmoid variational loss annealing schedule are used the
    BatchNorm will no longer be neccessary.
    """

    def __init__(self,
                 num_tokens,
                 max_output_length,
                 embedding_dimension=196,
                 filter_sizes=[9, 9, 10],
                 kernel_sizes=[9, 9, 11],
                 decoder_dimension=488,
                 **kwargs):
        """
        Parameters
        ----------
        filter_sizes: list of int
            Number of filters for each 1D convolution in the encoder
        kernel_sizes: list of int
            Kernel size for each 1D convolution in the encoder
        decoder_dimension: int
            Number of channels for the GRU Decoder
        """
        if len(filter_sizes) != len(kernel_sizes):
            raise ValueError("Must have same number of layers and kernels")
        self._filter_sizes = filter_sizes
        self._kernel_sizes = kernel_sizes
        self._decoder_dimension = decoder_dimension
        super(AspuruGuzikAutoEncoder,
              self).__init__(input_tokens=num_tokens,
                             output_tokens=num_tokens,
                             max_output_length=max_output_length,
                             embedding_dimension=embedding_dimension,
                             variational=True,
                             reverse_input=False,
                             **kwargs)

    def _create_features(self):
        return Input(shape=(self._max_output_length, len(self._input_tokens)))

    def _create_encoder(self, n_layers, dropout):
        """Create the encoder as a tf.keras.Model."""
        input = self._create_features()
        gather_indices = Input(shape=(2,), dtype=tf.int32)
        prev_layer = input
        for i in range(len(self._filter_sizes)):
            filter_size = self._filter_sizes[i]
            kernel_size = self._kernel_sizes[i]
            if dropout > 0.0:
                prev_layer = Dropout(rate=dropout)(prev_layer)
            prev_layer = Conv1D(filters=filter_size,
                                kernel_size=kernel_size,
                                activation=tf.nn.relu)(prev_layer)
        prev_layer = Flatten()(prev_layer)
        prev_layer = Dense(self._decoder_dimension,
                           activation=tf.nn.relu)(prev_layer)
        prev_layer = BatchNormalization()(prev_layer)
        return tf.keras.Model(inputs=[input, gather_indices],
                              outputs=prev_layer)

    def _create_decoder(self, n_layers, dropout):
        """Create the decoder as a tf.keras.Model."""
        input = Input(shape=(self._embedding_dimension,))
        prev_layer = Dense(self._embedding_dimension,
                           activation=tf.nn.relu)(input)
        prev_layer = layers.Stack()(self._max_output_length * [prev_layer])
        for i in range(3):
            if dropout > 0.0:
                prev_layer = Dropout(dropout)(prev_layer)
            prev_layer = GRU(self._decoder_dimension,
                             return_sequences=True)(prev_layer)
        output = Dense(len(self._output_tokens),
                       activation=tf.nn.softmax)(prev_layer)
        return tf.keras.Model(inputs=input, outputs=output)

    def _create_input_array(self, sequences):
        return self._create_output_array(sequences)

