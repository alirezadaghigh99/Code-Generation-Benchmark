def _fetch_frame_label(
    class_label_name: str,
    timestamp: float,
    rectangles_dict: Dict[str, List[Dict[str, Any]]],
    interpolation_cutoff_threshold_sec: Optional[float] = None,
    progressible_labels: Optional[Collection[int]] = None,
    carry_over_keys: Optional[List[str]] = None,
    required_keys: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Returns object labels for the specified video frame timestamp.

    The result will retain the structure of `rectangles_dict`, but just ensure that the
    timestamp value is as requested.

    If `progressible_labels` are supplied, the `"progress"` field will be included. This
    field represents the 'normalized' amount of time that the class label has existed
    temporally. See
    tests/data/datasets/utils/test_video.py:test_fetch_frame_with_progress for examples.

    This fetching function can be used for (per-frame) video classification pipelines.

    Args:
        class_label_name: The field name in `rectangles_dict` that maps to the class
            label.
        timestamps: A list of timestamps to fectch label from.
        rectangles_dict: (See docstring at top of file.)
        interpolation_cutoff_threshold_sec: Threshold under which we allow
            interpolation. In some `rectangles_dict`s, the labels (within the
            same track) are so far apart (e.g. 10 seconds) that interpolation is
            non-sensical. Thus this value prevents unrelated labels from being
            interpolated.
        progressible_labels: Set of labels for which to calculate `"progress"` for the
            resulting bounding boxes. If None, no `"progress"` field will be included.
        carry_over_keys: A list of keywords that specifies which keys should be carried
            over from the previous rectangle during interpolation. Defaults to None.
        required_keys: A list of keywords that specifies which keywords need to be
            included in a new bounding_box in addition to the class_label_name. Defaults
            to None.

    Returns:
        Dict containing the labels, still indexable by track id.
    """

    ret = {}
    for track_label, track_rectangles in rectangles_dict.items():
        all_times = [a["timestamp"] for a in track_rectangles]
        if not (all_times) == sorted(all_times):
            raise RuntimeError("all_times should be sorted.")

        numpy_times = np.array(all_times)

        idx = np.searchsorted(numpy_times, timestamp, side="right")
        if _before(timestamp, numpy_times[0]) or _before(numpy_times[-1], timestamp):
            # The track doesn't exist or has ceased to exist.
            ret[track_label] = _make_fake_bbox(
                rectangles_dict, timestamp, progressible_labels, required_keys
            )
            continue

        # @before_idx must be positive. If np.searchsorted() returned 0, the above "if"
        # statement would have been triggered.
        before_idx = idx - 1
        assert before_idx >= 0, f"Before_idx should be positive, but got {before_idx}."
        before_time = numpy_times[before_idx]

        if idx == len(numpy_times):
            after_idx = before_idx
            after_time = before_time
        else:
            after_idx = idx
            after_time = numpy_times[after_idx]

        # Either box for interpolation is invisible.
        if (
            not track_rectangles[before_idx]["is_visible"]
            or not track_rectangles[after_idx]["is_visible"]
        ):
            # We make a fake annotation for invisible boxes.
            ret[track_label] = _make_fake_bbox(
                rectangles_dict, timestamp, progressible_labels
            )
            continue

        # Boxes for interpolation are too far away.
        if (
            interpolation_cutoff_threshold_sec is not None
            and after_time - before_time > interpolation_cutoff_threshold_sec
        ):
            ret[track_label] = _make_fake_bbox(
                rectangles_dict, timestamp, progressible_labels, required_keys
            )
            continue

        # pylint: disable=unbalanced-tuple-unpacking
        x0, x1, y0, y1 = _interpolate_bounding_box(
            track_rectangles[before_idx],
            track_rectangles[after_idx],
            timestamp - before_time,
            after_time - before_time,
        )
        new_label = {}
        if carry_over_keys is not None:
            new_label = {
                key: track_rectangles[before_idx][key]
                for key in carry_over_keys
                if key in track_rectangles[before_idx]
            }

        if required_keys is not None:
            for key in required_keys:
                new_label[key] = track_rectangles[before_idx].get(key, -1)
        # New label will have updated coordinates and timestamp.
        new_label["x0"] = x0
        new_label["x1"] = x1
        new_label["y0"] = y0
        new_label["y1"] = y1
        new_label["timestamp"] = timestamp

        if progressible_labels is not None:
            progress = -1.0
            if track_rectangles[before_idx][class_label_name] in progressible_labels:
                search_fn = functools.partial(
                    _search_for_label_edge_timestamp,
                    class_label_name,
                    track_rectangles,
                    before_idx,
                    interpolation_cutoff_threshold_sec=interpolation_cutoff_threshold_sec,
                )
                start_timestamp = search_fn(-1)
                end_timestamp = search_fn(+1)
                progress = (timestamp - start_timestamp) / (
                    end_timestamp - start_timestamp
                )
            new_label["progress"] = progress

        _assert_progress_repr(class_label_name, progressible_labels, new_label)
        ret[track_label] = new_label

    return ret

