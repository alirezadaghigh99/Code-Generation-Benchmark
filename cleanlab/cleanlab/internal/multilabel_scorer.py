def exponential_moving_average(
    s: np.ndarray,
    *,
    alpha: Optional[float] = None,
    axis: int = 1,
    **_,
) -> np.ndarray:
    r"""Exponential moving average (EMA) score aggregation function.

    For a score vector s = (s_1, ..., s_K) with K scores, the values
    are sorted in *descending* order and the exponential moving average
    of the last score is calculated, denoted as EMA_K according to the
    note below.

    Note
    ----

    The recursive formula for the EMA at step :math:`t = 2, ..., K` is:

    .. math::

        \text{EMA}_t = \alpha \cdot s_t + (1 - \alpha) \cdot \text{EMA}_{t-1}, \qquad 0 \leq \alpha \leq 1

    We set :math:`\text{EMA}_1 = s_1` as the largest score in the sorted vector s.

    :math:`\alpha` is the "forgetting factor" that gives more weight to the
    most recent scores, and successively less weight to the previous scores.

    Parameters
    ----------
    s :
        Scores to be transformed.

    alpha :
        Discount factor that determines the weight of the previous EMA score.
        Higher alpha means that the previous EMA score has a lower weight while
        the current score has a higher weight.

        Its value must be in the interval [0, 1].

        If alpha is None, it is set to 2 / (K + 1) where K is the number of scores.

    axis :
        Axis along which the scores are sorted.

    Returns
    -------
    s_ema :
        Exponential moving average score.

    Examples
    --------
    >>> from cleanlab.internal.multilabel_scorer import exponential_moving_average
    >>> import numpy as np
    >>> s = np.array([[0.1, 0.2, 0.3]])
    >>> exponential_moving_average(s, alpha=0.5)
    np.array([0.175])
    """
    K = s.shape[1]
    s_sorted = np.fliplr(np.sort(s, axis=axis))
    if alpha is None:
        # One conventional choice for alpha is 2/(K + 1), where K is the number of periods in the moving average.
        alpha = float(2 / (K + 1))
    if not (0 <= alpha <= 1):
        raise ValueError(f"alpha must be in the interval [0, 1], got {alpha}")
    s_T = s_sorted.T
    s_ema, s_next = s_T[0], s_T[1:]
    for s_i in s_next:
        s_ema = alpha * s_i + (1 - alpha) * s_ema
    return s_ema

def _get_split_generator(labels, cv):
    _, multilabel_ids = np.unique(labels, axis=0, return_inverse=True)
    split_generator = cv.split(X=multilabel_ids, y=multilabel_ids)
    return split_generator

class Aggregator:
    """Helper class for aggregating the label quality scores for each class into a single score for each datapoint.

    Parameters
    ----------
    method:
        The method to compute the label quality scores for each class.
        If passed as a callable, your function should take in a 1D array of K scores and return a single aggregated score.
        See `~cleanlab.internal.multilabel_scorer.exponential_moving_average` for an example of such a function.
        Alternatively, this can be a str value to specify a built-in function, possible values are the keys of the ``Aggregator``'s `possible_methods` attribute.

    kwargs:
        Additional keyword arguments to pass to the aggregation function when it is called.
    """

    possible_methods: Dict[str, Callable[..., np.ndarray]] = {
        "exponential_moving_average": exponential_moving_average,
        "softmin": softmin,
    }

    def __init__(self, method: Union[str, Callable], **kwargs):
        if isinstance(method, str):  # convert to callable
            if method in self.possible_methods:
                method = self.possible_methods[method]
            else:
                raise ValueError(
                    f"Invalid aggregation method specified: '{method}', must be one of the following: {list(self.possible_methods.keys())}"
                )

        self._validate_method(method)
        self.method = method
        self.kwargs = kwargs

    @staticmethod
    def _validate_method(method) -> None:
        if not callable(method):
            raise TypeError(f"Expected callable method, got {type(method)}")

    @staticmethod
    def _validate_scores(scores: np.ndarray) -> None:
        if not (isinstance(scores, np.ndarray) and scores.ndim == 2):
            raise ValueError(
                f"Expected 2D array for scores, got {type(scores)} with shape {scores.shape}"
            )

    def __call__(self, scores: np.ndarray, **kwargs) -> np.ndarray:
        """Returns the label quality scores for each datapoint based on the given label quality scores for each class.

        Parameters
        ----------
        scores:
            The label quality scores for each class.

        Returns
        -------
        aggregated_scores:
            A single label quality score for each datapoint.
        """
        self._validate_scores(scores)
        kwargs["axis"] = 1
        updated_kwargs = {**self.kwargs, **kwargs}
        return self.method(scores, **updated_kwargs)

    def __repr__(self):
        return f"Aggregator(method={self.method.__name__}, kwargs={self.kwargs})"

class MultilabelScorer:
    """Aggregates label quality scores across different classes to produce one score per example in multi-label classification tasks.

    Parameters
    ----------
    base_scorer:
        The method to compute the label quality scores for each class.

        See the documentation for the ClassLabelScorer enum for more details.

    aggregator:
        The method to aggregate the label quality scores for each class into a single score for each datapoint.

        Defaults to the EMA (exponential moving average) aggregator with forgetting factor ``alpha=0.8``.

        See the documentation for the Aggregator class for more details.

        See also
        --------
        exponential_moving_average

    strict:
        Flag for performing strict validation of the input data.
    """

    def __init__(
        self,
        base_scorer: ClassLabelScorer = ClassLabelScorer.SELF_CONFIDENCE,
        aggregator: Union[Aggregator, Callable] = Aggregator(exponential_moving_average, alpha=0.8),
        *,
        strict: bool = True,
    ):
        self.base_scorer = base_scorer
        if not isinstance(aggregator, Aggregator):
            self.aggregator = Aggregator(aggregator)
        else:
            self.aggregator = aggregator
        self.strict = strict

    def __call__(
        self,
        labels: np.ndarray,
        pred_probs: np.ndarray,
        base_scorer_kwargs: Optional[dict] = None,
        **aggregator_kwargs,
    ) -> np.ndarray:
        """
        Computes a quality score for each label in a multi-label classification problem
        based on out-of-sample predicted probabilities.
        For each example, the label quality scores for each class are aggregated into a single overall label quality score.

        Parameters
        ----------
        labels:
            A 2D array of shape (n_samples, n_labels) with binary labels.

        pred_probs:
            A 2D array of shape (n_samples, n_labels) with predicted probabilities.

        kwargs:
            Additional keyword arguments to pass to the base_scorer and the aggregator.

        base_scorer_kwargs:
             Keyword arguments to pass to the base_scorer

         aggregator_kwargs:
             Additional keyword arguments to pass to the aggregator.

        Returns
        -------
        scores:
            A 1D array of shape (n_samples,) with the quality scores for each datapoint.

        Examples
        --------
        >>> from cleanlab.internal.multilabel_scorer import MultilabelScorer, ClassLabelScorer
        >>> import numpy as np
        >>> labels = np.array([[0, 1, 0], [1, 0, 1]])
        >>> pred_probs = np.array([[0.1, 0.9, 0.1], [0.4, 0.1, 0.9]])
        >>> scorer = MultilabelScorer()
        >>> scores = scorer(labels, pred_probs)
        >>> scores
        array([0.9, 0.5])

        >>> scorer = MultilabelScorer(
        ...     base_scorer = ClassLabelScorer.NORMALIZED_MARGIN,
        ...     aggregator = np.min,  # Use the "worst" label quality score for each example.
        ... )
        >>> scores = scorer(labels, pred_probs)
        >>> scores
        array([0.9, 0.4])
        """
        if self.strict:
            self._validate_labels_and_pred_probs(labels, pred_probs)
        scores = self.get_class_label_quality_scores(labels, pred_probs, base_scorer_kwargs)
        return self.aggregate(scores, **aggregator_kwargs)

    def aggregate(
        self,
        class_label_quality_scores: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Aggregates the label quality scores for each class into a single overall label quality score for each example.

        Parameters
        ----------
        class_label_quality_scores:
            A 2D array of shape (n_samples, n_labels) with the label quality scores for each class.

            See also
            --------
            get_class_label_quality_scores

        kwargs:
            Additional keyword arguments to pass to the aggregator.

        Returns
        -------
        scores:
            A 1D array of shape (n_samples,) with the quality scores for each datapoint.

        Examples
        --------
        >>> from cleanlab.internal.multilabel_scorer import MultilabelScorer
        >>> import numpy as np
        >>> class_label_quality_scores = np.array([[0.9, 0.9, 0.3],[0.4, 0.9, 0.6]])
        >>> scorer = MultilabelScorer() # Use the default aggregator (exponential moving average) with default parameters.
        >>> scores = scorer.aggregate(class_label_quality_scores)
        >>> scores
        array([0.42, 0.452])
        >>> new_scores = scorer.aggregate(class_label_quality_scores, alpha=0.5) # Use the default aggregator with custom parameters.
        >>> new_scores
        array([0.6, 0.575])

        Warning
        -------
        Make sure that keyword arguments correspond to the aggregation function used.
        I.e. the ``exponential_moving_average`` function supports an ``alpha`` keyword argument, but ``np.min`` does not.
        """
        return self.aggregator(class_label_quality_scores, **kwargs)

    def get_class_label_quality_scores(
        self,
        labels: np.ndarray,
        pred_probs: np.ndarray,
        base_scorer_kwargs: Optional[dict] = None,
    ) -> np.ndarray:
        """Computes separate label quality scores for each class.

        Parameters
        ----------
        labels:
            A 2D array of shape (n_samples, n_labels) with binary labels.

        pred_probs:
            A 2D array of shape (n_samples, n_labels) with predicted probabilities.

        base_scorer_kwargs:
            Keyword arguments to pass to the base scoring-function.

        Returns
        -------
        class_label_quality_scores:
            A 2D array of shape (n_samples, n_labels) with the quality scores for each label.

        Examples
        --------
        >>> from cleanlab.internal.multilabel_scorer import MultilabelScorer
        >>> import numpy as np
        >>> labels = np.array([[0, 1, 0], [1, 0, 1]])
        >>> pred_probs = np.array([[0.1, 0.9, 0.7], [0.4, 0.1, 0.6]])
        >>> scorer = MultilabelScorer() # Use the default base scorer (SELF_CONFIDENCE)
        >>> class_label_quality_scores = scorer.get_label_quality_scores_per_class(labels, pred_probs)
        >>> class_label_quality_scores
        array([[0.9, 0.9, 0.3],
               [0.4, 0.9, 0.6]])
        """
        class_label_quality_scores = np.zeros(shape=labels.shape)
        if base_scorer_kwargs is None:
            base_scorer_kwargs = {}
        for i, (label_i, pred_prob_i) in enumerate(zip(labels.T, pred_probs.T)):
            pred_prob_i_two_columns = stack_complement(pred_prob_i)
            class_label_quality_scores[:, i] = self.base_scorer(
                label_i, pred_prob_i_two_columns, **base_scorer_kwargs
            )
        return class_label_quality_scores

    @staticmethod
    def _validate_labels_and_pred_probs(labels: np.ndarray, pred_probs: np.ndarray) -> None:
        """
        Checks that (multi-)labels are in the proper binary indicator format and that
        they are compatible with the predicted probabilities.
        """
        # Only allow dense matrices for labels for now
        if not isinstance(labels, np.ndarray):
            raise TypeError("Labels must be a numpy array.")
        if not _is_multilabel(labels):
            raise ValueError("Labels must be in multi-label format.")
        if labels.shape != pred_probs.shape:
            raise ValueError("Labels and predicted probabilities must have the same shape.")

