class Classification(_HasID, Label):
    """A classification label.

    Args:
        label (None): the label string
        confidence (None): a confidence in ``[0, 1]`` for the classification
        logits (None): logits associated with the labels
    """

    label = fof.StringField()
    confidence = fof.FloatField()
    logits = fof.VectorField()

class Detection(_HasAttributesDict, _HasID, Label):
    """An object detection.

    Args:
        label (None): the label string
        bounding_box (None): a list of relative bounding box coordinates in
            ``[0, 1]`` in the following format::

            [<top-left-x>, <top-left-y>, <width>, <height>]

        mask (None): an instance segmentation mask for the detection within
            its bounding box, which should be a 2D binary or 0/1 integer numpy
            array
        confidence (None): a confidence in ``[0, 1]`` for the detection
        index (None): an index for the object
        attributes ({}): a dict mapping attribute names to :class:`Attribute`
            instances
    """

    label = fof.StringField()
    bounding_box = fof.ListField(fof.FloatField())
    mask = fof.ArrayField()
    confidence = fof.FloatField()
    index = fof.IntField()

    def to_polyline(self, tolerance=2, filled=True):
        """Returns a :class:`Polyline` representation of this instance.

        If the detection has a mask, the returned polyline will trace the
        boundary of the mask; otherwise, the polyline will trace the bounding
        box itself.

        Args:
            tolerance (2): a tolerance, in pixels, when generating an
                approximate polyline for the instance mask. Typical values are
                1-3 pixels
            filled (True): whether the polyline should be filled

        Returns:
            a :class:`Polyline`
        """
        dobj = foue.to_detected_object(self, extra_attrs=False)
        polyline = etai.convert_object_to_polygon(
            dobj, tolerance=tolerance, filled=filled
        )

        attributes = dict(self.iter_attributes())

        return Polyline(
            label=self.label,
            points=polyline.points,
            confidence=self.confidence,
            index=self.index,
            closed=polyline.closed,
            filled=polyline.filled,
            tags=self.tags,
            **attributes,
        )

    def to_segmentation(self, mask=None, frame_size=None, target=255):
        """Returns a :class:`Segmentation` representation of this instance.

        The detection must have an instance mask, i.e., its :attr:`mask`
        attribute must be populated.

        You must provide either ``mask`` or ``frame_size`` to use this method.

        Args:
            mask (None): an optional numpy array to use as an initial mask to
                which to add this object
            frame_size (None): the ``(width, height)`` of the segmentation
                mask to render. This parameter has no effect if a ``mask`` is
                provided
            target (255): the pixel value or RGB hex string to use to render
                the object

        Returns:
            a :class:`Segmentation`
        """
        if self.mask is None:
            raise ValueError(
                "Only detections with their `mask` attributes populated can "
                "be converted to segmentations"
            )

        mask, target = _parse_segmentation_target(mask, frame_size, target)
        _render_instance(mask, self, target)
        return Segmentation(mask=mask)

    def to_shapely(self, frame_size=None):
        """Returns a Shapely representation of this instance.

        Args:
            frame_size (None): the ``(width, height)`` of the image. If
                provided, the returned geometry will use absolute coordinates

        Returns:
            a ``shapely.geometry.polygon.Polygon``
        """
        # pylint: disable=unpacking-non-sequence
        x, y, w, h = self.bounding_box

        if frame_size is not None:
            width, height = frame_size
            x *= width
            y *= height
            w *= width
            h *= height

        return sg.box(x, y, x + w, y + h)

    @classmethod
    def from_mask(cls, mask, label=None, **attributes):
        """Creates a :class:`Detection` instance with its ``mask`` attribute
        populated from the given full image mask.

        The instance mask for the object is extracted by computing the bounding
        rectangle of the non-zero values in the image mask.

        Args:
            mask: a boolean or 0/1 numpy array
            label (None): the label string
            **attributes: additional attributes for the :class:`Detection`

        Returns:
            a :class:`Detection`
        """
        if mask.ndim > 2:
            mask = mask[:, :, 0]

        bbox, mask = _parse_stuff_instance(mask.astype(bool))

        return cls(label=label, bounding_box=bbox, mask=mask, **attributes)

class Regression(_HasID, Label):
    """A regression value.

    Args:
        value (None): the regression value
        confidence (None): a confidence in ``[0, 1]`` for the regression
    """

    value = fof.FloatField()
    confidence = fof.FloatField()

class Detections(_HasLabelList, Label):
    """A list of object detections in an image.

    Args:
        detections (None): a list of :class:`Detection` instances
    """

    _LABEL_LIST_FIELD = "detections"

    detections = fof.ListField(fof.EmbeddedDocumentField(Detection))

    def to_polylines(self, tolerance=2, filled=True):
        """Returns a :class:`Polylines` representation of this instance.

        For detections with masks, the returned polylines will trace the
        boundaries of the masks; otherwise, the polylines will trace the
        bounding boxes themselves.

        Args:
            tolerance (2): a tolerance, in pixels, when generating approximate
                polylines for the instance masks
            filled (True): whether the polylines should be filled

        Returns:
            a :class:`Polylines`
        """
        # pylint: disable=not-an-iterable
        return Polylines(
            polylines=[
                d.to_polyline(tolerance=tolerance, filled=filled)
                for d in self.detections
            ]
        )

    def to_segmentation(self, mask=None, frame_size=None, mask_targets=None):
        """Returns a :class:`Segmentation` representation of this instance.

        Only detections with instance masks (i.e., their :attr:`mask`
        attributes populated) will be rendered.

        You must provide either ``mask`` or ``frame_size`` to use this method.

        Args:
            mask (None): an optional array to use as an initial mask to which
                to add objects
            frame_size (None): the ``(width, height)`` of the segmentation
                mask to render. This parameter has no effect if a ``mask`` is
                provided
            mask_targets (None): a dict mapping integer pixel values (2D masks)
                or RGB hex strings (3D masks) to label strings defining which
                object classes to render and which pixel values to use for each
                class. If omitted, all objects are rendered with pixel value
                255

        Returns:
            a :class:`Segmentation`
        """
        mask, labels_to_targets = _parse_segmentation_mask_targets(
            mask, frame_size, mask_targets
        )

        # pylint: disable=not-an-iterable
        for detection in self.detections:
            if detection.mask is None:
                msg = "Skipping detection(s) with no instance mask"
                warnings.warn(msg)
                continue

            if labels_to_targets is not None:
                target = labels_to_targets.get(detection.label, None)
                if target is None:
                    continue  # skip unknown target
            else:
                target = 255

            _render_instance(mask, detection, target)

        return Segmentation(mask=mask)

class Classifications(_HasLabelList, Label):
    """A list of classifications for an image.

    Args:
        classifications (None): a list of :class:`Classification` instances
        logits (None): logits associated with the labels
    """

    _LABEL_LIST_FIELD = "classifications"

    classifications = fof.ListField(fof.EmbeddedDocumentField(Classification))
    logits = fof.VectorField()

class Polyline(_HasAttributesDict, _HasID, Label):
    """A set of semantically related polylines or polygons.

    Args:
        label (None): a label for the polyline
        points (None): a list of lists of ``(x, y)`` points in
            ``[0, 1] x [0, 1]`` describing the vertices of each shape in the
            polyline
        confidence (None): a confidence in ``[0, 1]`` for the polyline
        index (None): an index for the polyline
        closed (False): whether the shapes are closed, i.e., and edge should
            be drawn from the last vertex to the first vertex of each shape
        filled (False): whether the polyline represents polygons, i.e., shapes
            that should be filled when rendering them
        attributes ({}): a dict mapping attribute names to :class:`Attribute`
            instances for the polyline
    """

    label = fof.StringField()
    points = fof.PolylinePointsField()
    confidence = fof.FloatField()
    index = fof.IntField()
    closed = fof.BooleanField(default=False)
    filled = fof.BooleanField(default=False)

    def to_detection(self, mask_size=None, frame_size=None):
        """Returns a :class:`Detection` representation of this instance whose
        bounding box tightly encloses the polyline.

        If a ``mask_size`` is provided, an instance mask of the specified size
        encoding the polyline's shape is included.

        Alternatively, if a ``frame_size`` is provided, the required mask size
        is then computed based off of the polyline points and ``frame_size``.

        Args:
            mask_size (None): an optional ``(width, height)`` at which to
                render an instance mask for the polyline
            frame_size (None): used when no ``mask_size`` is provided.
                an optional ``(width, height)`` of the frame containing this
                polyline that is used to compute the required ``mask_size``

        Returns:
            a :class:`Detection`
        """
        polyline = foue.to_polyline(self, extra_attrs=False)
        if mask_size is not None:
            bbox, mask = etai.render_bounding_box_and_mask(polyline, mask_size)
        else:
            bbox = etai.render_bounding_box(polyline)
            mask = None

        xtl, ytl, xbr, ybr = bbox.to_coords()
        bounding_box = [xtl, ytl, (xbr - xtl), (ybr - ytl)]

        if mask_size is None and frame_size:
            w, h = frame_size
            rel_mask_w = bounding_box[2]
            rel_mask_h = bounding_box[3]
            abs_mask_w = int(round(rel_mask_w * w))
            abs_mask_h = int(round(rel_mask_h * h))
            mask_size = (abs_mask_w, abs_mask_h)
            _, mask = etai.render_bounding_box_and_mask(polyline, mask_size)

        attributes = dict(self.iter_attributes())

        return Detection(
            label=self.label,
            bounding_box=bounding_box,
            confidence=self.confidence,
            mask=mask,
            index=self.index,
            tags=self.tags,
            **attributes,
        )

    def to_segmentation(
        self, mask=None, frame_size=None, target=255, thickness=1
    ):
        """Returns a :class:`Segmentation` representation of this instance.

        You must provide either ``mask`` or ``frame_size`` to use this method.

        Args:
            mask (None): an optional numpy array to use as an initial mask to
                which to add objects
            frame_size (None): the ``(width, height)`` of the segmentation
                mask to render. This parameter has no effect if a ``mask`` is
                provided
            target (255): the pixel value or RGB hex string to use to render
                the object
            thickness (1): the thickness, in pixels, at which to render
                (non-filled) polylines

        Returns:
            a :class:`Segmentation`
        """
        mask, target = _parse_segmentation_target(mask, frame_size, target)
        _render_polyline(mask, self, target, thickness)
        return Segmentation(mask=mask)

    def to_shapely(self, frame_size=None, filled=None):
        """Returns a Shapely representation of this instance.

        The type of geometry returned depends on the number of shapes
        (:attr:`points`) and whether they are polygons or lines
        (:attr:`filled`).

        Args:
            frame_size (None): the ``(width, height)`` of the image. If
                provided, the returned geometry will use absolute coordinates
            filled (None): whether to treat the shape as filled (True) or
                hollow (False) regardless of its :attr:`filled` attribute

        Returns:
            one of the following:

            -   ``shapely.geometry.polygon.Polygon``: if :attr:`filled` is True
                and :attr:`points` contains a single shape
            -   ``shapely.geometry.multipolygon.MultiPolygon``: if
                :attr:`filled` is True and :attr:`points` contains multiple
                shapes
            -   ``shapely.geometry.linestring.LineString``: if :attr:`filled`
                is False and :attr:`points` contains a single shape
            -   ``shapely.geometry.multilinestring.MultiLineString``: if
                :attr:`filled` is False and :attr:`points` contains multiple
                shapes
        """
        if filled is not None:
            _filled = filled
        else:
            _filled = self.filled

        if self.closed:
            points = []
            for shape in self.points:  # pylint: disable=not-an-iterable
                if shape:
                    shape = list(shape) + [shape[0]]

                points.append(shape)
        else:
            points = self.points

        if frame_size is not None:
            w, h = frame_size
            points = [[(x * w, y * h) for x, y in shape] for shape in points]

        if len(points) == 1:
            if _filled:
                return sg.Polygon(points[0])

            return sg.LineString(points[0])

        if _filled:
            return sg.MultiPolygon(list(zip(points, itertools.repeat(None))))

        return sg.MultiLineString(points)

    @classmethod
    def from_mask(cls, mask, label=None, tolerance=2, **attributes):
        """Creates a :class:`Polyline` instance with polygons describing the
        non-zero region(s) of the given full image mask.

        Args:
            mask: a boolean or 0/1 numpy array
            label (None): the label string
            tolerance (2): a tolerance, in pixels, when generating approximate
                polygons for each region. Typical values are 1-3 pixels
            **attributes: additional attributes for the :class:`Polyline`

        Returns:
            a :class:`Polyline`
        """
        if mask.ndim > 2:
            mask = mask[:, :, 0]

        points = _get_polygons(
            mask.astype(bool),
            tolerance=tolerance,
        )

        return cls(
            label=label, points=points, filled=True, closed=True, **attributes
        )

    @classmethod
    def from_cuboid(cls, vertices, frame_size=None, label=None, **attributes):
        """Constructs a cuboid from its 8 vertices in the format below::

               7--------6
              /|       /|
             / |      / |
            3--------2  |
            |  4-----|--5
            | /      | /
            |/       |/
            0--------1

        If a ``frame_size`` is provided, ``vertices`` must be absolute pixel
        coordinates; otherwise ``vertices`` should be normalized coordinates in
        ``[0, 1] x [0, 1]``.

        Args:
            vertices: a list of 8 ``(x, y)`` vertices in the above format
            frame_size (None): the ``(width, height)`` of the frame
            label (None): the label string
            **attributes: additional arguments for the :class:`Polyline`

        Returns:
            a :class:`Polyline`
        """
        vertices = np.asarray(vertices)
        if frame_size is not None:
            vertices /= np.asarray(frame_size)[np.newaxis, :]

        front = vertices[:4]
        back = vertices[4:]
        top = vertices[[3, 2, 6, 7], :]
        bottom = vertices[[0, 1, 5, 4], :]
        faces = [front.tolist(), back.tolist(), top.tolist(), bottom.tolist()]
        return cls(label=label, points=faces, closed=True, **attributes)

    @classmethod
    def from_rotated_box(
        cls, xc, yc, w, h, theta, frame_size=None, label=None, **attributes
    ):
        """Constructs a rotated bounding box from its center, dimensions, and
        rotation.

        If a ``frame_size`` is provided, the provided box coordinates must be
        absolute pixel coordinates; otherwise they should be normalized
        coordinates in ``[0, 1]``. Note that rotations in normalized
        coordinates only make sense when the source aspect ratio is square.

        Args:
            xc: the x-center coordinate
            yc: the y-center coorindate
            w: the box width
            y: the box height
            theta: the counter-clockwise rotation of the box in radians
            frame_size (None): the ``(width, height)`` of the frame
            label (None): the label string
            **attributes: additional arguments for the :class:`Polyline`

        Returns:
            a :class:`Polyline`
        """
        R = _rotation_matrix(theta)
        x = 0.5 * w * np.array([1, -1, -1, 1])
        y = 0.5 * h * np.array([1, 1, -1, -1])
        points = R.dot(np.stack((x, y))).T + np.array((xc, yc))
        if frame_size is not None:
            points /= np.asarray(frame_size)[np.newaxis, :]

        points = points.tolist()
        return cls(label=label, points=[points], closed=True, **attributes)

class Polylines(_HasLabelList, Label):
    """A list of polylines or polygons in an image.

    Args:
        polylines (None): a list of :class:`Polyline` instances
    """

    _LABEL_LIST_FIELD = "polylines"

    polylines = fof.ListField(fof.EmbeddedDocumentField(Polyline))

    def to_detections(self, mask_size=None, frame_size=None):
        """Returns a :class:`Detections` representation of this instance whose
        bounding boxes tightly enclose the polylines.

        If a ``mask_size`` is provided, instance masks of the specified size
        encoding the polyline's shape are included in each :class:`Detection`.

        Alternatively, if a ``frame_size`` is provided, the required mask size
        is then computed based off of the polyline points and ``frame_size``.

        Args:
            mask_size (None): an optional ``(width, height)`` at which to
                render instance masks for the polylines
            frame_size (None): used when no ``mask_size`` is provided.
                an optional ``(width, height)`` of the frame containing these
                polylines that is used to compute the required ``mask_size``

        Returns:
            a :class:`Detections`
        """
        # pylint: disable=not-an-iterable
        return Detections(
            detections=[
                p.to_detection(mask_size=mask_size, frame_size=frame_size)
                for p in self.polylines
            ]
        )

    def to_segmentation(
        self, mask=None, frame_size=None, mask_targets=None, thickness=1
    ):
        """Returns a :class:`Segmentation` representation of this instance.

        You must provide either ``mask`` or ``frame_size`` to use this method.

        Args:
            mask (None): an optional numpy array to use as an initial mask to
                which to add objects
            frame_size (None): the ``(width, height)`` of the segmentation
                mask to render. This parameter has no effect if a ``mask`` is
                provided
            mask_targets (None): a dict mapping integer pixel values (2D masks)
                or RGB hex strings (3D masks) to label strings defining which
                object classes to render and which pixel values to use for each
                class. If omitted, all objects are rendered with pixel value
                255
            thickness (1): the thickness, in pixels, at which to render
                (non-filled) polylines

        Returns:
            a :class:`Segmentation`
        """
        mask, labels_to_targets = _parse_segmentation_mask_targets(
            mask, frame_size, mask_targets
        )

        # pylint: disable=not-an-iterable
        for polyline in self.polylines:
            if labels_to_targets is not None:
                target = labels_to_targets.get(polyline.label, None)
                if target is None:
                    continue  # skip unknown target
            else:
                target = 255

            _render_polyline(mask, polyline, target, thickness)

        return Segmentation(mask=mask)

class Keypoint(_HasAttributesDict, _HasID, Label):
    """A list of keypoints in an image.

    Args:
        label (None): a label for the points
        points (None): a list of ``(x, y)`` keypoints in ``[0, 1] x [0, 1]``
        confidence (None): a list of confidences in ``[0, 1]`` for each point
        index (None): an index for the keypoints
        attributes ({}): a dict mapping attribute names to :class:`Attribute`
            instances
    """

    label = fof.StringField()
    points = fof.KeypointsField()
    confidence = fof.ListField(fof.FloatField(), null=True)
    index = fof.IntField()

    def to_shapely(self, frame_size=None):
        """Returns a Shapely representation of this instance.

        Args:
            frame_size (None): the ``(width, height)`` of the image. If
                provided, the returned geometry will use absolute coordinates

        Returns:
            a ``shapely.geometry.multipoint.MultiPoint``
        """
        # pylint: disable=not-an-iterable
        points = self.points

        if frame_size is not None:
            w, h = frame_size
            points = [(x * w, y * h) for x, y in points]

        return sg.MultiPoint(points)

class Keypoints(_HasLabelList, Label):
    """A list of :class:`Keypoint` instances in an image.

    Args:
        keypoints (None): a list of :class:`Keypoint` instances
    """

    _LABEL_LIST_FIELD = "keypoints"

    keypoints = fof.ListField(fof.EmbeddedDocumentField(Keypoint))

