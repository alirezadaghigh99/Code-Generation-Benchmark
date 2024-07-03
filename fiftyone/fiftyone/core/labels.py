class Classification(_HasID, Label):
    """A classification label.

    Args:
        label (None): the label string
        confidence (None): a confidence in ``[0, 1]`` for the classification
        logits (None): logits associated with the labels
    """

    label = fof.StringField()
    confidence = fof.FloatField()
    logits = fof.VectorField()class Detection(_HasAttributesDict, _HasID, Label):
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