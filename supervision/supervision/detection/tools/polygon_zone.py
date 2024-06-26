class PolygonZone:
    """
    A class for defining a polygon-shaped zone within a frame for detecting objects.

    Attributes:
        polygon (np.ndarray): A polygon represented by a numpy array of shape
            `(N, 2)`, containing the `x`, `y` coordinates of the points.
        triggering_anchors (Iterable[sv.Position]): A list of positions specifying
            which anchors of the detections bounding box to consider when deciding on
            whether the detection fits within the PolygonZone
            (default: (sv.Position.BOTTOM_CENTER,)).
        current_count (int): The current count of detected objects within the zone
        mask (np.ndarray): The 2D bool mask for the polygon zone
    """

    @deprecated_parameter(
        old_parameter="triggering_position",
        new_parameter="triggering_anchors",
        map_function=lambda x: [x],
        warning_message="`{old_parameter}` in `{function_name}` is deprecated and will "
        "be remove in `supervision-0.23.0`. Use '{new_parameter}' "
        "instead.",
    )
    def __init__(
        self,
        polygon: npt.NDArray[np.int64],
        frame_resolution_wh: Optional[Tuple[int, int]] = None,
        triggering_anchors: Iterable[Position] = (Position.BOTTOM_CENTER,),
    ):
        if frame_resolution_wh is not None:
            warnings.warn(
                "The `frame_resolution_wh` parameter is no longer required and will be "
                "dropped in version supervision-0.24.0. The mask resolution is now "
                "calculated automatically based on the polygon coordinates.",
                category=SupervisionWarnings,
            )

        self.polygon = polygon.astype(int)
        self.triggering_anchors = triggering_anchors
        if not list(self.triggering_anchors):
            raise ValueError("Triggering anchors cannot be empty.")

        self.current_count = 0

        x_max, y_max = np.max(polygon, axis=0)
        self.frame_resolution_wh = (x_max + 1, y_max + 1)
        self.mask = polygon_to_mask(
            polygon=polygon, resolution_wh=(x_max + 2, y_max + 2)
        )

    def trigger(self, detections: Detections) -> npt.NDArray[np.bool_]:
        """
        Determines if the detections are within the polygon zone.

        Parameters:
            detections (Detections): The detections
                to be checked against the polygon zone

        Returns:
            np.ndarray: A boolean numpy array indicating
                if each detection is within the polygon zone
        """

        clipped_xyxy = clip_boxes(
            xyxy=detections.xyxy, resolution_wh=self.frame_resolution_wh
        )
        clipped_detections = replace(detections, xyxy=clipped_xyxy)
        all_clipped_anchors = np.array(
            [
                np.ceil(clipped_detections.get_anchors_coordinates(anchor)).astype(int)
                for anchor in self.triggering_anchors
            ]
        )

        is_in_zone: npt.NDArray[np.bool_] = (
            self.mask[all_clipped_anchors[:, :, 1], all_clipped_anchors[:, :, 0]]
            .transpose()
            .astype(bool)
        )

        is_in_zone: npt.NDArray[np.bool_] = np.all(is_in_zone, axis=1)
        self.current_count = int(np.sum(is_in_zone))
        return is_in_zone.astype(bool)