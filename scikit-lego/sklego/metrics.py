def subset_score(subset_picker: Callable, score: Callable, **kwargs):
    """Return a method that applies the passed score only to a specific subset.

    The subset picker is a method that is passed the corresponding `X` and `y_true` and returns a one-dimensional
    boolean vector where every element corresponds to a row in the data.

    Only the elements with a True value are taken into account for the passed score, representing a filter.

    This allows users to have an easy approach to measuring metrics over different slices of the population which can
    give insights into the model performance, either specifically for fairness or in general.

    Parameters
    ----------
    subset_picker : Callable
        Method that returns a boolean mask that is used for slicing the samples
    score : Callable[..., T]
        The score that needs to be applied to the subset
    kwargs : dict
        Additional keyword arguments to pass to score

    Returns
    -------
    Callable[..., T]
        A function which calculates the passed score for the subset

    Examples
    --------
    ```py
    from sklego.metrics import subset_score
    ...
    subset_score(lambda X, y_true: X['column'] == 'A', accuracy_score)(clf, X, y)
    ```
    """

    def sliced_metric(estimator, X, y_true=None):
        mask = subset_picker(X, y_true)
        if isinstance(mask, np.ndarray):
            if len(mask.shape) > 1:
                raise ValueError(
                    "`subset_picker` should return 1-dimensional numpy array or Pandas"
                    + " series, returned {} instead".format(len(mask.shape))
                )
        if np.sum(mask) == 0:
            warnings.warn("No samples in subset, returning NaN", RuntimeWarning)
            return np.nan
        X = X[mask]
        y_pred = estimator.predict(X)
        return score(y_true[mask], y_pred, **kwargs)

    return sliced_metric

def p_percent_score(sensitive_column, positive_target=1):
    r"""The p_percent score calculates the ratio between the probability of a positive outcome given the sensitive
    attribute (column) being true and the same probability given the sensitive attribute being false.

    $$\min \left(\frac{P(\hat{y}=1 | z=1)}{P(\hat{y}=1 | z=0)}, \frac{P(\hat{y}=1 | z=0)}{P(\hat{y}=1 | z=1)}\right)$$

    This is especially useful to use in situations where "fairness" is a theme.

    Parameters
    ----------
    sensitive_column : str | int
        Name of the column containing the binary sensitive attribute (when X is a dataframe) or the index of the column
        (when X is a numpy array).
    positive_target : int, default=1
        The name of the class which is associated with a positive outcome

    Returns
    -------
    Callable[..., float]
        A function which calculates the p percent score for z = column

    Examples
    --------
    ```py
    from sklego.metrics import p_percent_score
    ...
    p_percent_score('gender')(clf, X, y)
    ```

    Source
    ------
    M. Zafar et al. (2017), Fairness Constraints: Mechanisms for Fair Classification
    """

    def impl(estimator, X, y_true=None):
        """Remember: X is the thing going *in* to your pipeline."""
        sensitive_col = X[:, sensitive_column] if isinstance(X, np.ndarray) else X[sensitive_column]

        if not ((sensitive_col == 0) | (sensitive_col == 1)).all():
            raise ValueError(
                f"p_percent_score only supports binary indicator columns for `column`. "
                f"Found values {np.unique(sensitive_col)}"
            )

        y_hat = estimator.predict(X)
        y_given_z1 = y_hat[sensitive_col == 1]
        y_given_z0 = y_hat[sensitive_col == 0]
        p_y1_z1 = np.mean(y_given_z1 == positive_target)
        p_y1_z0 = np.mean(y_given_z0 == positive_target)

        # If we never predict a positive target for one of the subgroups, the model is by definition not
        # fair so we return 0
        if p_y1_z1 == 0:
            warnings.warn(
                f"No samples with y_hat == {positive_target} for {sensitive_column} == 1, returning 0",
                RuntimeWarning,
            )
            return 0

        if p_y1_z0 == 0:
            warnings.warn(
                f"No samples with y_hat == {positive_target} for {sensitive_column} == 0, returning 0",
                RuntimeWarning,
            )
            return 0

        p_percent = np.minimum(p_y1_z1 / p_y1_z0, p_y1_z0 / p_y1_z1)
        return p_percent if not np.isnan(p_percent) else 1

    return impl

