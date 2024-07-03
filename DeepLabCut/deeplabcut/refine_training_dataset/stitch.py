class Tracklet:
    def __init__(self, data, inds):
        """
        Create a Tracklet object.

        Parameters
        ----------
        data : ndarray
            3D array of shape (nframes, nbodyparts, 3 or 4), where the last
            dimension is for x, y, likelihood and, optionally, identity.
        inds : array-like
            Corresponding time frame indices.
        """
        if data.ndim != 3 or data.shape[-1] not in (3, 4):
            raise ValueError("Data must of shape (nframes, nbodyparts, 3 or 4)")

        if data.shape[0] != len(inds):
            raise ValueError(
                "Data and corresponding indices must have the same length."
            )

        self.data = data.astype(np.float64)
        self.inds = np.array(inds)
        monotonically_increasing = all(a < b for a, b in zip(inds, inds[1:]))
        if not monotonically_increasing:
            idx = np.argsort(inds, kind="mergesort")  # For stable sort with duplicates
            self.inds = self.inds[idx]
            self.data = self.data[idx]
        self._centroid = None

    def __len__(self):
        return self.inds.size

    def __add__(self, other):
        """Join this tracklet to another one."""
        data = np.concatenate((self.data, other.data))
        inds = np.concatenate((self.inds, other.inds))
        return Tracklet(data, inds)

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def __sub__(self, other):
        mask = np.isin(self.inds, other.inds, assume_unique=True)
        if mask.all():
            return None
        return Tracklet(self.data[~mask], self.inds[~mask])

    def __lt__(self, other):
        """Test whether this tracklet precedes the other one."""
        return self.end < other.start

    def __gt__(self, other):
        """Test whether this tracklet follows the other one."""
        return self.start > other.end

    def __contains__(self, other_tracklet):
        """Test whether tracklets temporally overlap."""
        return np.isin(self.inds, other_tracklet.inds, assume_unique=True).any()

    def __repr__(self):
        return (
            f"Tracklet of length {len(self)} from {self.start} to {self.end} "
            f"with reliability {self.likelihood:.3f}"
        )

    @property
    def xy(self):
        """Return the x and y coordinates."""
        return self.data[..., :2]

    @property
    def centroid(self):
        """
        Return the instantaneous 2D position of the Tracklet centroid.
        For Tracklets longer than 10 frames, the centroid is automatically
        smoothed using an exponential moving average.
        The result is cached for efficiency.
        """
        if self._centroid is None:
            self._update_centroid()
        return self._centroid

    def _update_centroid(self):
        like = self.data[..., 2:3]
        self._centroid = np.nansum(self.xy * like, axis=1) / np.nansum(like, axis=1)

    @property
    def likelihood(self):
        """Return the average likelihood of all Tracklet detections."""
        return np.nanmean(self.data[..., 2])

    @property
    def identity(self):
        """Return the average predicted identity of all Tracklet detections."""
        try:
            return mode(
                self.data[..., 3],
                axis=None,
                nan_policy="omit",
                keepdims=False,
            )[0]
        except IndexError:
            return -1

    @property
    def start(self):
        """Return the time at which the tracklet starts."""
        return self.inds[0]

    @property
    def end(self):
        """Return the time at which the tracklet ends."""
        return self.inds[-1]

    @property
    def flat_data(self):
        return self.data[..., :3].reshape((len(self)), -1)

    def get_data_at(self, ind):
        return self.data[np.searchsorted(self.inds, ind)]

    def set_data_at(self, ind, data):
        self.data[np.searchsorted(self.inds, ind)] = data

    def del_data_at(self, ind):
        idx = np.searchsorted(self.inds, ind)
        self.inds = np.delete(self.inds, idx)
        self.data = np.delete(self.data, idx, axis=0)
        self._update_centroid()

    def interpolate(self, max_gap=1):
        if max_gap < 1:
            raise ValueError("Gap should be a strictly positive integer.")

        gaps = np.diff(self.inds) - 1
        valid_gaps = (0 < gaps) & (gaps <= max_gap)
        fills = []
        for i in np.flatnonzero(valid_gaps):
            s, e = self.inds[[i, i + 1]]
            data1, data2 = self.data[[i, i + 1]]
            diff = (data2 - data1) / (e - s)
            diff[np.isnan(diff)] = 0
            interp = diff[..., np.newaxis] * np.arange(1, e - s)
            data = data1 + np.rollaxis(interp, axis=2)
            data[..., 2] = 0.5  # Chance detections
            if data.shape[1] == 4:
                data[:, 3] = self.identity
            fills.append(Tracklet(data, np.arange(s + 1, e)))
        if not fills:
            return self
        return self + sum(fills)

    def contains_duplicates(self, return_indices=False):
        """
        Evaluate whether the Tracklet contains duplicate time indices.
        If `return_indices`, also return the indices of the duplicates.
        """
        has_duplicates = len(set(self.inds)) != len(self.inds)
        if not return_indices:
            return has_duplicates
        return has_duplicates, np.flatnonzero(np.diff(self.inds) == 0)

    def calc_velocity(self, where="head", norm=True):
        """
        Calculate the linear velocity of either the `head`
        or `tail` of the Tracklet, computed over the last or first
        three frames, respectively. If `norm`, return the absolute
        speed rather than a 2D vector.
        """
        if where == "tail":
            vel = (
                np.diff(self.centroid[:3], axis=0)
                / np.diff(self.inds[:3])[:, np.newaxis]
            )
        elif where == "head":
            vel = (
                np.diff(self.centroid[-3:], axis=0)
                / np.diff(self.inds[-3:])[:, np.newaxis]
            )
        else:
            raise ValueError(f"Unknown where={where}")
        if norm:
            return np.sqrt(np.sum(vel ** 2, axis=1)).mean()
        return vel.mean(axis=0)

    @property
    def maximal_velocity(self):
        vel = np.diff(self.centroid, axis=0) / np.diff(self.inds)[:, np.newaxis]
        return np.sqrt(np.max(np.sum(vel ** 2, axis=1)))

    def calc_rate_of_turn(self, where="head"):
        """
        Calculate the rate of turn (or angular velocity) of
        either the `head` or `tail` of the Tracklet, computed over
        the last or first three frames, respectively.
        """
        if where == "tail":
            v = np.diff(self.centroid[:3], axis=0)
        else:
            v = np.diff(self.centroid[-3:], axis=0)
        theta = np.arctan2(v[:, 1], v[:, 0])
        return (theta[1] - theta[0]) / (self.inds[1] - self.inds[0])

    @property
    def is_continuous(self):
        """Test whether there are gaps in the time indices."""
        return self.end - self.start + 1 == len(self)

    def immediately_follows(self, other_tracklet, max_gap=1):
        """
        Test whether this Tracklet follows another within
        a tolerance of`max_gap` frames.
        """
        return 0 < self.start - other_tracklet.end <= max_gap

    def distance_to(self, other_tracklet):
        """
        Calculate the Euclidean distance between this Tracklet and another.
        If the Tracklets overlap in time, this is the mean distance over
        those frames. Otherwise, it is the distance between the head/tail
        of one to the tail/head of the other.
        """
        if self in other_tracklet:
            dist = (
                self.centroid[np.isin(self.inds, other_tracklet.inds)]
                - other_tracklet.centroid[np.isin(other_tracklet.inds, self.inds)]
            )
            return np.sqrt(np.sum(dist ** 2, axis=1)).mean()
        elif self < other_tracklet:
            return np.sqrt(
                np.sum((self.centroid[-1] - other_tracklet.centroid[0]) ** 2)
            )
        else:
            return np.sqrt(
                np.sum((self.centroid[0] - other_tracklet.centroid[-1]) ** 2)
            )

    def motion_affinity_with(self, other_tracklet):
        """
        Evaluate the motion affinity of this Tracklet' with another one.
        This evaluates whether the Tracklets could realistically be reached
        by one another, knowing the time separating them and their velocities.
        Return 0 if the Tracklets overlap.
        """
        time_gap = self.time_gap_to(other_tracklet)
        if time_gap > 0:
            if self < other_tracklet:
                d1 = self.centroid[-1] + time_gap * self.calc_velocity(norm=False)
                d2 = other_tracklet.centroid[
                    0
                ] - time_gap * other_tracklet.calc_velocity("tail", False)
                delta1 = other_tracklet.centroid[0] - d1
                delta2 = self.centroid[-1] - d2
            else:
                d1 = other_tracklet.centroid[
                    -1
                ] + time_gap * other_tracklet.calc_velocity(norm=False)
                d2 = self.centroid[0] - time_gap * self.calc_velocity("tail", False)
                delta1 = self.centroid[0] - d1
                delta2 = other_tracklet.centroid[-1] - d2
            return (np.sqrt(np.sum(delta1 ** 2)) + np.sqrt(np.sum(delta2 ** 2))) / 2
        return 0

    def time_gap_to(self, other_tracklet):
        """Return the time gap separating this Tracklet to another."""
        if self in other_tracklet:
            t = 0
        elif self < other_tracklet:
            t = other_tracklet.start - self.end
        else:
            t = self.start - other_tracklet.end
        return t

    def shape_dissimilarity_with(self, other_tracklet):
        """Calculate the dissimilarity in shape between this Tracklet and another."""
        if self in other_tracklet:
            dist = np.inf
        elif self < other_tracklet:
            dist = self.undirected_hausdorff(self.xy[-1], other_tracklet.xy[0])
        else:
            dist = self.undirected_hausdorff(self.xy[0], other_tracklet.xy[-1])
        return dist

    def box_overlap_with(self, other_tracklet):
        """Calculate the overlap between each Tracklet's bounding box."""
        if self in other_tracklet:
            overlap = 0
        else:
            if self < other_tracklet:
                bbox1 = self.calc_bbox(-1)
                bbox2 = other_tracklet.calc_bbox(0)
            else:
                bbox1 = self.calc_bbox(0)
                bbox2 = other_tracklet.calc_bbox(-1)
            overlap = calc_iou(bbox1, bbox2)
        return overlap

    @staticmethod
    def undirected_hausdorff(u, v):
        return max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])

    def calc_bbox(self, ind):
        xy = self.xy[ind]
        bbox = np.empty(4)
        bbox[:2] = np.nanmin(xy, axis=0)
        bbox[2:] = np.nanmax(xy, axis=0)
        return bbox

    @staticmethod
    def hankelize(xy):
        ncols = int(np.ceil(len(xy) * 2 / 3))
        nrows = len(xy) - ncols + 1
        mat = np.empty((2 * nrows, ncols))
        mat[::2] = hankel(xy[:nrows, 0], xy[-ncols:, 0])
        mat[1::2] = hankel(xy[:nrows, 1], xy[-ncols:, 1])
        return mat

    def to_hankelet(self):
        # See Li et al., 2012. Cross-view Activity Recognition using Hankelets.
        # As proposed in the paper, the Hankel matrix can either be formed from
        # the tracklet's centroid or its normalized velocity.
        # vel = np.diff(self.centroid, axis=0)
        # vel /= np.linalg.norm(vel, axis=1, keepdims=True)
        # return self.hankelize(vel)
        return self.hankelize(self.centroid)

    def dynamic_dissimilarity_with(self, other_tracklet):
        """
        Compute a dissimilarity score between Hankelets.
        This metric efficiently captures the degree of alignment of
        the subspaces spanned by the columns of both matrices.

        See Li et al., 2012.
            Cross-view Activity Recognition using Hankelets.
        """
        hk1 = self.to_hankelet()
        hk1 /= np.linalg.norm(hk1)
        hk2 = other_tracklet.to_hankelet()
        hk2 /= np.linalg.norm(hk2)
        min_shape = min(hk1.shape + hk2.shape)
        temp1 = (hk1 @ hk1.T)[:min_shape, :min_shape]
        temp2 = (hk2 @ hk2.T)[:min_shape, :min_shape]
        return 2 - np.linalg.norm(temp1 + temp2)

    def dynamic_similarity_with(self, other_tracklet, tol=0.01):
        """
        Evaluate the complexity of the tracklets' underlying dynamics
        from the rank of their Hankel matrices, and assess whether
        they originate from the same track. The idea is that if two
        tracklets are part of the same track, they can be approximated
        by a low order regressor. Conversely, tracklets belonging to
        different tracks will require a higher order regressor.

        See Dicle et al., 2013.
            The Way They Move: Tracking Multiple Targets with Similar Appearance.
        """
        # TODO Add missing data imputation
        joint_tracklet = self + other_tracklet
        joint_rank = joint_tracklet.estimate_rank(tol)
        rank1 = self.estimate_rank(tol)
        rank2 = other_tracklet.estimate_rank(tol)
        return (rank1 + rank2) / joint_rank - 1

    def estimate_rank(self, tol):
        """
        Estimate the (low) rank of a noisy matrix via
        hard thresholding of singular values.

        See Gavish & Donoho, 2013.
            The optimal hard threshold for singular values is 4/sqrt(3)
        """
        mat = self.to_hankelet()
        # nrows, ncols = mat.shape
        # beta = nrows / ncols
        # omega = 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43
        _, s, _ = sli.svd(mat, min(10, min(mat.shape)))
        # return np.argmin(s > omega * np.median(s))
        eigen = s ** 2
        diff = np.abs(np.diff(eigen / eigen[0]))
        return np.argmin(diff > tol)

    def plot(self, centroid_only=True, color=None, ax=None, interactive=False):
        if ax is None:
            fig, ax = plt.subplots()
        centroid = np.full((self.end + 1, 2), np.nan)
        centroid[self.inds] = self.centroid
        lines = ax.plot(centroid, c=color, lw=2, picker=interactive)
        if not centroid_only:
            xy = np.full((self.end + 1, self.xy.shape[1], 2), np.nan)
            xy[self.inds] = self.xy
            ax.plot(xy[..., 0], c=color, lw=1)
            ax.plot(xy[..., 1], c=color, lw=1)
        return lines    def from_dict_of_dict(
        cls,
        dict_of_dict,
        n_tracks,
        min_length=10,
        split_tracklets=True,
        prestitch_residuals=True,
    ):
        tracklets = []
        header = dict_of_dict.pop("header", None)
        single = None
        for k, dict_ in dict_of_dict.items():
            try:
                inds, data = zip(*[(cls.get_frame_ind(k), v) for k, v in dict_.items()])
            except ValueError:
                continue
            inds = np.asarray(inds)
            data = np.asarray(data)
            try:
                nrows, ncols = data.shape
                data = data.reshape((nrows, ncols // 3, 3))
            except ValueError:
                pass
            tracklet = Tracklet(data, inds)
            if k == "single":
                single = tracklet
            else:
                tracklets.append(Tracklet(data, inds))
        class_ = cls(
            tracklets, n_tracks, min_length, split_tracklets, prestitch_residuals
        )
        class_.header = header
        class_.single = single
        return class_