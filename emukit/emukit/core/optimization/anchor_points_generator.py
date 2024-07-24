class ObjectiveAnchorPointsGenerator(AnchorPointsGenerator):
    """
    This anchor points generator chooses points where the acquisition function is highest
    """

    def __init__(self, space: ParameterSpace, acquisition: Acquisition, num_samples: int = 1000):
        """
        :param space: The parameter space describing the input domain of the non-context variables
        :param acquisition: The acquisition function
        :param num_samples: The number of points at which the anchor point scores are calculated
        """
        self.acquisition = acquisition
        super(ObjectiveAnchorPointsGenerator, self).__init__(space, num_samples)

    def get_anchor_point_scores(self, X: np.ndarray) -> np.ndarray:
        """
        :param X: The samples at which to evaluate the criterion
        :return:
        """
        are_constraints_satisfied = np.all(
            [np.ones(X.shape[0])] + [c.evaluate(X) for c in self.space.constraints], axis=0
        )
        scores = np.zeros((X.shape[0],))
        scores[~are_constraints_satisfied] = -np.inf
        scores[are_constraints_satisfied] = self.acquisition.evaluate(X[are_constraints_satisfied, :])[:, 0]
        return scores

