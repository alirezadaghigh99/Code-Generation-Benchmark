class KerasWrapperSequential:
    """Makes any ``tf.keras.models.Sequential`` object compatible with :py:class:`CleanLearning <cleanlab.classification.CleanLearning>` and sklearn.

    `KerasWrapperSequential` is instantiated in the same way as a keras ``Sequential``  object, except for optional extra `compile_kwargs` argument.
    Just instantiate this object in the same way as your ``tf.keras.models.Sequential`` object (rather than passing in an existing ``Sequential`` object).
    The instance methods of this class work in the same way as those of any keras ``Sequential`` object, see the `Keras documentation <https://keras.io/>`_ for details.

    Parameters
    ----------
    layers: list
        A list containing the layers to add to the keras ``Sequential`` model (same as for ``tf.keras.models.Sequential``).

    name: str, default = None
        Name for the Keras model (same as for ``tf.keras.models.Sequential``).

    compile_kwargs: dict, default = {"loss": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)}
        Dict of optional keyword arguments to pass into ``model.compile()`` for declaring loss, metrics, optimizer, etc.
    """

    def __init__(
        self,
        layers: Optional[list] = None,
        name: Optional[str] = None,
        compile_kwargs: dict = {
            "loss": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        },
        params: Optional[dict] = None,
    ):
        if params is None:
            params = {}

        self.layers = layers
        self.name = name
        self.compile_kwargs = compile_kwargs
        self.params = params
        self.net = None

    def get_params(self, deep=True):
        """Returns the parameters of the Keras model."""
        return {
            "layers": self.layers,
            "name": self.name,
            "compile_kwargs": self.compile_kwargs,
            "params": self.params,
        }

    def set_params(self, **params):
        """Set the parameters of the Keras model."""
        self.params.update(params)
        return self

    def fit(self, X, y=None, **kwargs):
        """Trains a Sequential Keras model.

        Parameters
        ----------
        X : tf.Dataset or np.array or pd.DataFrame
            If ``X`` is a tensorflow dataset object, it must already contain the labels as is required for standard Keras fit.

        y : np.array or pd.DataFrame, default = None
            If ``X`` is a tensorflow dataset object, you can optionally provide the labels again here as argument `y` to be compatible with sklearn,
            but they are ignored.
            If ``X`` is a numpy array or pandas dataframe, the labels have to be passed in using this argument.
        """
        if self.net is None:
            self.net = tf.keras.models.Sequential(self.layers, self.name)
            self.net.compile(**self.compile_kwargs)

        # TODO: check for generators
        if y is not None and not isinstance(X, (tf.data.Dataset, keras.utils.Sequence)):
            kwargs["y"] = y

        self.net.fit(X, **{**self.params, **kwargs})

    def predict_proba(self, X, *, apply_softmax=True, **kwargs):
        """Predict class probabilities for all classes using the wrapped Keras model.
        Set extra argument `apply_softmax` to True to indicate your network only outputs logits not probabilities.

        Parameters
        ----------
        X : tf.Dataset or np.array or pd.DataFrame
            Data in the same format as the original ``X`` provided to ``fit()``.
        """
        if self.net is None:
            raise ValueError("must call fit() before predict()")
        pred_probs = self.net.predict(X, **kwargs)
        if apply_softmax:
            pred_probs = tf.nn.softmax(pred_probs, axis=1)
        return pred_probs

    def predict(self, X, **kwargs):
        """Predict class labels using the wrapped Keras model.

        Parameters
        ----------
        X : tf.Dataset or np.array or pd.DataFrame
            Data in the same format as the original ``X`` provided to ``fit()``.
        """
        pred_probs = self.predict_proba(X, **kwargs)
        return np.argmax(pred_probs, axis=1)

    def summary(self, **kwargs):
        """Returns the summary of the Keras model."""
        if self.net is None:
            self.net = tf.keras.models.Sequential(self.layers, self.name)
            self.net.compile(**self.compile_kwargs)

        return self.net.summary(**kwargs)

