class KeypointSkeleton(EmbeddedDocument):
    """Description of a keypoint skeleton.

    Keypoint skeletons can be associated with
    :class:`fiftyone.core.labels.Keypoint` or
    :class:`fiftyone.core.labels.Keypoints` fields whose
    :attr:`points <fiftyone.core.labels.Keypoint.points>` attributes all
    contain a fixed number of semantically ordered points.

    The ``edges`` argument contains lists of integer indexes that define the
    connectivity of the points in the skeleton, and the optional ``labels``
    argument defines the label strings for each node in the skeleton.

    For example, the skeleton below is defined by edges between the following
    nodes::

        left hand <-> left shoulder <-> right shoulder <-> right hand
        left eye <-> right eye <-> mouth

    Example::

        import fiftyone as fo

        # A skeleton for an object made of 7 points
        skeleton = fo.KeypointSkeleton(
            labels=[
                "left hand" "left shoulder", "right shoulder", "right hand",
                "left eye", "right eye", "mouth",
            ],
            edges=[[0, 1, 2, 3], [4, 5, 6]],
        )

    Args:
        labels (None): an optional list of label strings for each node
        edges: a list of lists of integer indexes defining the connectivity
            between nodes
    """

    # strict=False lets this class ignore unknown fields from other versions
    meta = {"strict": False}

    labels = ListField(StringField(), null=True)
    edges = ListField(ListField(IntField()))

class ColorScheme(EmbeddedDocument):
    """Description of a color scheme in the App.

    Example::

        import fiftyone as fo
        import fiftyone.zoo as foz

        dataset = foz.load_zoo_dataset("quickstart")

        # Store a custom color scheme for a dataset
        dataset.app_config.color_scheme = fo.ColorScheme(
            color_by="field",
            color_pool=[
                "#ff0000",
                "#00ff00",
                "#0000ff",
                "pink",
                "yellowgreen",
            ],
            fields=[
                {
                    "path": "ground_truth",
                    "fieldColor": "#ff00ff",
                    "colorByAttribute": "label",
                    "valueColors": [{"value": "dog", "color": "yellow"}],
                    "maskTargetsColors": [
                        {"intTarget": 2, "color": "#ff0000"},
                        {"intTarget": 12, "color": "#99ff00"},
                    ],
                }
            ],
            label_tags={
                "fieldColor": "#00ffff",
                "valueColors": [
                    {"value": "correct", "color": "#ff00ff"},
                    {"value": "mistake", "color": "#00ff00"},
                ],
            },
            colorscales=[
                {
                    "path": "heatmap1",
                    "list": [
                        {"value": 0, "color": "rgb(0, 0, 255)"},
                        {"value": 1, "color": "rgb(0, 255, 255)"},
                    ],
                },
                {
                    "path": "heatmap2",
                    "name": "hsv",
                },
            ],
            multicolor_keypoints=False,
            opacity=0.5,
            show_skeletons=True,
            default_mask_targets_colors=[
                {"intTarget": 1, "color": "#FEC0AA"},
                {"intTarget": 2, "color": "#EC4E20"},
            ],
            default_colorscale={"name": "sunset", "list": None},
        )

        session = fo.launch_app(dataset)

    Args:
        color_by (None): whether annotations should be colored by ``"field"``,
            ``"value"``, or ``"instance"``
        color_pool (None): an optional list of colors to use as a color pool
            for this dataset
        multicolor_keypoints (None): whether to use multiple colors for
            keypoints
        opacity (None): transparency of the annotation, between 0 and 1
        show_skeletons (None): whether to show skeletons of keypoints
        fields (None): an optional list of dicts of per-field custom colors
            with the following keys:

            -   ``path`` (required): the fully-qualified path to the field
                you're customizing
            -   ``fieldColor`` (optional): a color to assign to the field in
                the App sidebar
            -   ``colorByAttribute`` (optional): the attribute to use to assign
                per-value colors. Only applicable when the field is an embedded
                document
            -   ``valueColors`` (optional): a list of dicts specifying colors
                to use for individual values of this field
            -   ``maskTargetsColors`` (optional): a list of dicts specifying
                index and color for 2D masks
        default_mask_targets_colors (None): a list of dicts with the following
            keys specifying index and color for 2D masks of the dataset. If a
            field does not have field specific mask targets colors, this list
            will be used:

            -   ``intTarget``: integer target value
            -   ``color``: a color string
        default_colorscale (None): dataset default colorscale dict with the
            following keys:

            -   ``name`` (optional): a named plotly colorscale, e.g. ``"hsv"``.
                See https://plotly.com/python/builtin-colorscales
            -   ``list`` (optional): a list of dicts of colorscale values

                -   ``value``: a float number between 0 and 1. A valid list
                    must have have colors defined for 0 and 1
                -   ``color``: an rgb color string
        colorscales (None): an optional list of dicts of per-field custom
            colorscales with the following keys:

            -   ``path`` (required): the fully-qualified path to the field
                you're customizing. use "dataset" if you are setting the
                default colorscale for dataset
            -   ``name`` (optional): a named colorscale plotly recognizes
            -   ``list`` (optional): a list of dicts of colorscale values with
                the following keys:

                -   ``value``: a float number between 0 and 1. A valid list
                    must have have colors defined for 0 and 1
                -   ``color``: an rgb color string
        label_tags (None): an optional dict specifying custom colors for label
            tags with the following keys:

            -   ``fieldColor`` (optional): a color to assign to all label tags
            -   ``valueColors`` (optional): a list of dicts
    """

    # strict=False lets this class ignore unknown fields from other versions
    meta = {"strict": False}

    id = ObjectIdField(
        required=True,
        default=lambda: str(ObjectId()),
        db_field="_id",
    )
    color_pool = ListField(ColorField(), null=True)
    color_by = StringField(null=True)
    fields = ListField(DictField(), null=True)
    label_tags = DictField(null=True)
    multicolor_keypoints = BooleanField(null=True)
    opacity = FloatField(null=True)
    show_skeletons = BooleanField(null=True)
    default_mask_targets_colors = ListField(DictField(), null=True)
    colorscales = ListField(DictField(), null=True)
    default_colorscale = DictField(null=True)

    @property
    def _id(self):
        return ObjectId(self.id)

    @_id.setter
    def _id(self, value):
        self.id = str(value)

