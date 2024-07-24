def check_class_weight_balanced_linear_classifier(name, Classifier):
    """Test class weights with non-contiguous class labels."""
    # this is run on classes, not instances, though this should be changed
    X = np.array([[-1.0, -1.0], [-1.0, 0], [-0.8, -1.0], [1.0, 1.0], [1.0, 0.0]])
    y = np.array([1, 1, 1, -1, -1])

    classifier = Classifier()

    if hasattr(classifier, "n_iter"):
        # This is a very small dataset, default n_iter are likely to prevent
        # convergence
        classifier.set_params(n_iter=1000)
    if hasattr(classifier, "max_iter"):
        classifier.set_params(max_iter=1000)
    if hasattr(classifier, "cv"):
        classifier.set_params(cv=3)
    set_random_state(classifier)

    # Let the model compute the class frequencies
    classifier.set_params(class_weight="balanced")
    coef_balanced = classifier.fit(X, y).coef_.copy()

    # Count each label occurrence to reweight manually
    n_samples = len(y)
    n_classes = float(len(np.unique(y)))

    class_weight = {
        1: n_samples / (np.sum(y == 1) * n_classes),
        -1: n_samples / (np.sum(y == -1) * n_classes),
    }
    classifier.set_params(class_weight=class_weight)
    coef_manual = classifier.fit(X, y).coef_.copy()

    assert_allclose(
        coef_balanced,
        coef_manual,
        err_msg="Classifier %s is not computing class_weight=balanced properly." % name,
    )

class _NotAnArray:
    """An object that is convertible to an array.

    Parameters
    ----------
    data : array-like
        The data.
    """

    def __init__(self, data):
        self.data = np.asarray(data)

    def __array__(self, dtype=None, copy=None):
        return self.data

    def __array_function__(self, func, types, args, kwargs):
        if func.__name__ == "may_share_memory":
            return True
        raise TypeError("Don't want to call array_function {}!".format(func.__name__))

