class PlanarityFilter(object):
    """
    compute planarity and return false if the planarity of a pointcloud is above or below a threshold

    Parameters
    ----------
    thresh: float, optional
        threshold to filter low planar pointcloud
    is_leq: bool, optional
        choose whether planarity should be lesser or equal than the threshold or greater than the threshold.
    """

    def __init__(self, thresh=0.3, is_leq=True):
        self.thresh = thresh
        self.is_leq = is_leq

    def __call__(self, data):
        if getattr(data, "eigenvalues", None) is None:
            data = PCACompute()(data)
        planarity = compute_planarity(data.eigenvalues)
        if self.is_leq:
            return planarity <= self.thresh
        else:
            return planarity > self.thresh

    def __repr__(self):
        return "{}(thresh={}, is_leq={})".format(self.__class__.__name__, self.thresh, self.is_leq)

