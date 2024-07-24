class DeltaFScore(StoppingCriterion):
    """A stopping criterion which stops if the predicted change of the F-score falls below
    a threshold [AB19]_.

    .. note:: This criterion is only applicable for binary classification.

    .. versionchanged:: 1.3.3
       The implementation now correctly only considers the change in agreement of the predicted labels
       belonging to the positive class.
    """
    def __init__(self, num_classes, window_size=3, threshold=0.05):
        """
        num_classes : int
            Number of classes.
        window_size : int, default=3
            Defines number of iterations for which the predictions are taken into account, i.e.
            this stopping criterion only sees the last `window_size`-many states of the prediction
            array passed to `stop()`.
        threshold : float, default=0.05
            The criterion stops when the predicted F-score falls below this threshold.
        """
        self.num_classes = num_classes

        if num_classes != 2:
            raise ValueError('DeltaFScore is only applicable for binary classifications '
                             '(requires num_class=2)')

        self.window_size = window_size
        self.threshold = threshold

        self.last_predictions = None
        self.delta_history = []

    def stop(self, active_learner=None, predictions=None, proba=None, indices_stopping=None):
        check_window_based_predictions(predictions, self.last_predictions)

        if self.last_predictions is None:
            self.last_predictions = predictions
            return False
        else:
            agreement = ((self.last_predictions == 1) & (predictions == 1)).astype(int).sum()
            disagreement_old_positive = ((self.last_predictions == 1) & (predictions == 0)).astype(int).sum()
            disagreement_new_positive = ((self.last_predictions == 0) & (predictions == 1)).astype(int).sum()

            denominator = (2 * agreement + disagreement_old_positive + disagreement_new_positive)
            delta_f = 1 - 2 * agreement / denominator

            self.delta_history.append(delta_f)
            self.last_predictions = predictions

            if len(self.delta_history) < self.window_size:
                return False

            self.delta_history = self.delta_history[-self.window_size:]

            if np.all(np.array(self.delta_history) < self.threshold):
                return True
            else:
                return False

