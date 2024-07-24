def calc_bboxes_from_keypoints(data, slack=0, offset=0):
    data = np.asarray(data)
    if data.shape[-1] < 3:
        raise ValueError("Data should be of shape (n_animals, n_bodyparts, 3)")

    if data.ndim != 3:
        data = np.expand_dims(data, axis=0)
    bboxes = np.full((data.shape[0], 5), np.nan)
    bboxes[:, :2] = np.nanmin(data[..., :2], axis=1) - slack  # X1, Y1
    bboxes[:, 2:4] = np.nanmax(data[..., :2], axis=1) + slack  # X2, Y2
    bboxes[:, -1] = np.nanmean(data[..., 2])  # Average confidence
    bboxes[:, [0, 2]] += offset
    return bboxes

class Ellipse:
    def __init__(self, x, y, width, height, theta):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.theta = theta  # in radians
        self._geometry = None

    @property
    def parameters(self):
        return self.x, self.y, self.width, self.height, self.theta

    @property
    def aspect_ratio(self):
        return max(self.width, self.height) / min(self.width, self.height)

    def calc_similarity_with(self, other_ellipse):
        max_dist = max(
            self.height, self.width, other_ellipse.height, other_ellipse.width
        )
        dist = math.sqrt(
            (self.x - other_ellipse.x) ** 2 + (self.y - other_ellipse.y) ** 2
        )
        cost1 = 1 - min(dist / max_dist, 1)
        cost2 = abs(math.cos(self.theta - other_ellipse.theta))
        return 0.8 * cost1 + 0.2 * cost2 * cost1

    def contains_points(self, xy, tol=0.1):
        ca = math.cos(self.theta)
        sa = math.sin(self.theta)
        x_demean = xy[:, 0] - self.x
        y_demean = xy[:, 1] - self.y
        return (
            ((ca * x_demean + sa * y_demean) ** 2 / (0.5 * self.width) ** 2)
            + ((sa * x_demean - ca * y_demean) ** 2 / (0.5 * self.height) ** 2)
        ) <= 1 + tol

    def draw(self, show_axes=True, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.transforms import Affine2D

        if ax is None:
            ax = plt.subplot(111, aspect="equal")
        el = patches.Ellipse(
            xy=(self.x, self.y),
            width=self.width,
            height=self.height,
            angle=np.rad2deg(self.theta),
            **kwargs,
        )
        ax.add_patch(el)
        if show_axes:
            major = Line2D([-self.width / 2, self.width / 2], [0, 0], lw=3, zorder=3)
            minor = Line2D([0, 0], [-self.height / 2, self.height / 2], lw=3, zorder=3)
            trans = (
                Affine2D().rotate(self.theta).translate(self.x, self.y) + ax.transData
            )
            major.set_transform(trans)
            minor.set_transform(trans)
            ax.add_artist(major)
            ax.add_artist(minor)

class EllipseFitter:
    def __init__(self, sd=2):
        self.sd = sd
        self.x = None
        self.y = None
        self.params = None
        self._coeffs = None

    def fit(self, xy):
        self.x, self.y = xy[np.isfinite(xy).all(axis=1)].T
        if len(self.x) < 3:
            return None
        if self.sd:
            self.params = self._fit_error(self.x, self.y, self.sd)
        else:
            self._coeffs = self._fit(self.x, self.y)
            self.params = self.calc_parameters(self._coeffs)
        if not np.isnan(self.params).any():
            el = Ellipse(*self.params)
            # Regularize by forcing AR <= 5
            # max_ar = 5
            # if el.aspect_ratio >= max_ar:
            #     if el.height > el.width:
            #         el.width = el.height / max_ar
            #     else:
            #         el.height = el.width / max_ar
            # Orient the ellipse such that it encompasses most points
            # n_inside = el.contains_points(np.c_[self.x, self.y]).sum()
            # el.theta += 0.5 * np.pi
            # if el.contains_points(np.c_[self.x, self.y]).sum() < n_inside:
            #     el.theta -= 0.5 * np.pi
            return el
        return None

    @staticmethod
    @jit(nopython=True)
    def _fit(x, y):
        """
        Least Squares ellipse fitting algorithm
        Fit an ellipse to a set of X- and Y-coordinates.
        See Halir and Flusser, 1998 for implementation details

        :param x: ndarray, 1D trajectory
        :param y: ndarray, 1D trajectory
        :return: 1D ndarray of 6 coefficients of the general quadratic curve:
            ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0
        """
        D1 = np.vstack((x * x, x * y, y * y))
        D2 = np.vstack((x, y, np.ones_like(x)))
        S1 = D1 @ D1.T
        S2 = D1 @ D2.T
        S3 = D2 @ D2.T
        T = -np.linalg.inv(S3) @ S2.T
        temp = S1 + S2 @ T
        M = np.zeros_like(temp)
        M[0] = temp[2] * 0.5
        M[1] = -temp[1]
        M[2] = temp[0] * 0.5
        E, V = np.linalg.eig(M)
        cond = 4 * V[0] * V[2] - V[1] ** 2
        a1 = V[:, cond > 0][:, 0]
        a2 = T @ a1
        return np.hstack((a1, a2))

    @staticmethod
    @jit(nopython=True)
    def _fit_error(x, y, sd):
        """
        Fit a sd-sigma covariance error ellipse to the data.

        :param x: ndarray, 1D input of X coordinates
        :param y: ndarray, 1D input of Y coordinates
        :param sd: int, size of the error ellipse in 'standard deviation'
        :return: ellipse center, semi-axes length, angle to the X-axis
        """
        cov = np.cov(x, y)
        E, V = np.linalg.eigh(cov)  # Returns the eigenvalues in ascending order
        # r2 = chi2.ppf(2 * norm.cdf(sd) - 1, 2)
        # height, width = np.sqrt(E * r2)
        height, width = 2 * sd * np.sqrt(E)
        a, b = V[:, 1]
        rotation = math.atan2(b, a) % np.pi
        return [np.mean(x), np.mean(y), width, height, rotation]

    @staticmethod
    @jit(nopython=True)
    def calc_parameters(coeffs):
        """
        Calculate ellipse center coordinates, semi-axes lengths, and
        the counterclockwise angle of rotation from the x-axis to the ellipse major axis.
        Visit http://mathworld.wolfram.com/Ellipse.html
        for how to estimate ellipse parameters.

        :param coeffs: list of fitting coefficients
        :return: center: 1D ndarray, semi-axes: 1D ndarray, angle: float
        """
        # The general quadratic curve has the form:
        # ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0
        a, b, c, d, f, g = coeffs
        b *= 0.5
        d *= 0.5
        f *= 0.5

        # Ellipse center coordinates
        x0 = (c * d - b * f) / (b * b - a * c)
        y0 = (a * f - b * d) / (b * b - a * c)

        # Semi-axes lengths
        num = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
        den1 = (b * b - a * c) * (np.sqrt((a - c) ** 2 + 4 * b * b) - (a + c))
        den2 = (b * b - a * c) * (-np.sqrt((a - c) ** 2 + 4 * b * b) - (a + c))
        major = np.sqrt(num / den1)
        minor = np.sqrt(num / den2)

        # Angle to the horizontal
        if b == 0:
            if a < c:
                phi = 0
            else:
                phi = np.pi / 2
        else:
            if a < c:
                phi = np.arctan(2 * b / (a - c)) / 2
            else:
                phi = np.pi / 2 + np.arctan(2 * b / (a - c)) / 2

        return [x0, y0, 2 * major, 2 * minor, phi]

class EllipseTracker(BaseTracker):
    def __init__(self, params):
        super().__init__(dim=5, dim_z=5)
        self.kf.R[2:, 2:] *= 10.0
        # High uncertainty to the unobservable initial velocities
        self.kf.P[5:, 5:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[5:, 5:] *= 0.01
        self.state = params

    @BaseTracker.state.setter
    def state(self, params):
        state = np.asarray(params).reshape((-1, 1))
        super(EllipseTracker, type(self)).state.fset(self, state)

class BoxTracker(BaseTracker):
    def __init__(self, bbox):
        super().__init__(dim=4, dim_z=4)
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )
        self.kf.R[2:, 2:] *= 10.0
        # Give high uncertainty to the unobservable initial velocities
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.state = bbox

    def update(self, bbox):
        super().update(self.convert_bbox_to_z(bbox))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        return super().predict()

    @property
    def state(self):
        return self.convert_x_to_bbox(self.kf.x)[0]

    @state.setter
    def state(self, bbox):
        state = self.convert_bbox_to_z(bbox)
        super(BoxTracker, type(self)).state.fset(self, state)

    @staticmethod
    def convert_x_to_bbox(x, score=None):
        """
        Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
        [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
        """
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if score is None:
            return np.array(
                [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]
            ).reshape((1, 4))
        else:
            return np.array(
                [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]
            ).reshape((1, 5))

    @staticmethod
    def convert_bbox_to_z(bbox):
        """
        Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
        [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
        the aspect ratio
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h  # scale is just area
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

class SORTEllipse(SORTBase):
    def __init__(self, max_age, min_hits, iou_threshold, sd=2):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.fitter = EllipseFitter(sd)
        EllipseTracker.n_trackers = 0
        super().__init__()

    def track(self, poses, identities=None):
        self.n_frames += 1

        trackers = np.zeros((len(self.trackers), 6))
        for i in range(len(trackers)):
            trackers[i, :5] = self.trackers[i].predict()
        empty = np.isnan(trackers).any(axis=1)
        trackers = trackers[~empty]
        for ind in np.flatnonzero(empty)[::-1]:
            self.trackers.pop(ind)

        ellipses = []
        pred_ids = []
        for i, pose in enumerate(poses):
            el = self.fitter.fit(pose)
            if el is not None:
                ellipses.append(el)
                if identities is not None:
                    pred_ids.append(mode(identities[i])[0][0])
        if not len(trackers):
            matches = np.empty((0, 2), dtype=int)
            unmatched_detections = np.arange(len(ellipses))
            unmatched_trackers = np.empty((0, 6), dtype=int)
        else:
            ellipses_trackers = [Ellipse(*t[:5]) for t in trackers]
            cost_matrix = np.zeros((len(ellipses), len(ellipses_trackers)))
            for i, el in enumerate(ellipses):
                for j, el_track in enumerate(ellipses_trackers):
                    cost = el.calc_similarity_with(el_track)
                    if identities is not None:
                        match = 2 if pred_ids[i] == self.trackers[j].id_ else 1
                        cost *= match
                    cost_matrix[i, j] = cost
            row_indices, col_indices = linear_sum_assignment(cost_matrix, maximize=True)
            unmatched_detections = [
                i for i, _ in enumerate(ellipses) if i not in row_indices
            ]
            unmatched_trackers = [
                j for j, _ in enumerate(trackers) if j not in col_indices
            ]
            matches = []
            for row, col in zip(row_indices, col_indices):
                val = cost_matrix[row, col]
                # diff = val - cost_matrix
                # diff[row, col] += val
                # if (
                #         val < self.iou_threshold
                #         or np.any(diff[row] <= 0.2)
                #         or np.any(diff[:, col] <= 0.2)
                # ):
                if val < self.iou_threshold:
                    unmatched_detections.append(row)
                    unmatched_trackers.append(col)
                else:
                    matches.append([row, col])
            if not len(matches):
                matches = np.empty((0, 2), dtype=int)
            else:
                matches = np.stack(matches)
            unmatched_trackers = np.asarray(unmatched_trackers)
            unmatched_detections = np.asarray(unmatched_detections)

        animalindex = []
        for t, tracker in enumerate(self.trackers):
            if t not in unmatched_trackers:
                ind = matches[matches[:, 1] == t, 0][0]
                animalindex.append(ind)
                tracker.update(ellipses[ind].parameters)
            else:
                animalindex.append(-1)

        for i in unmatched_detections:
            trk = EllipseTracker(ellipses[i].parameters)
            if identities is not None:
                trk.id_ = mode(identities[i])[0][0]
            self.trackers.append(trk)
            animalindex.append(i)

        i = len(self.trackers)
        ret = []
        for trk in reversed(self.trackers):
            d = trk.state
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.n_frames <= self.min_hits
            ):
                ret.append(
                    np.concatenate((d, [trk.id, int(animalindex[i - 1])])).reshape(
                        1, -1
                    )
                )  # for DLC we also return the original animalid
                # +1 as MOT benchmark requires positive >> this is removed for DLC!
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 7))

