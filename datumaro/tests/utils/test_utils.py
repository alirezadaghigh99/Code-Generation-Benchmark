def compare_datasets(
    test,
    expected: IDataset,
    actual: IDataset,
    ignored_attrs: Union[None, Literal["*"], Collection[str]] = None,
    require_media: bool = False,
    ignore_ann_id: bool = False,
    ignore_ann_group: bool = False,
    **kwargs,
):
    compare_categories(test, expected.categories(), actual.categories())

    test.assertTrue(issubclass(actual.media_type(), expected.media_type()))

    test.assertEqual(sorted(expected.subsets()), sorted(actual.subsets()))
    test.assertEqual(len(expected), len(actual))

    for item_a in expected:
        item_b = find(actual, lambda x: x.id == item_a.id and x.subset == item_a.subset)
        test.assertFalse(item_b is None, item_a.id)

        if ignored_attrs and ignored_attrs != IGNORE_ALL:
            test.assertEqual(
                item_a.attributes,
                filter_dict(item_b.attributes, exclude_keys=ignored_attrs),
                item_a.id,
            )
        elif not ignored_attrs:
            test.assertEqual(item_a.attributes, item_b.attributes, item_a.id)

        if require_media and item_a.media and item_b.media:
            if isinstance(item_a.media, VideoFrame):
                test.assertEqual(item_a.media, item_b.media, item_a.id)
                test.assertEqual(item_a.media.index, item_b.media.index, item_a.id)
            elif isinstance(item_a.media, Image):
                test.assertEqual(item_a.media, item_b.media, item_a.id)
            elif isinstance(item_a.media, PointCloud):
                test.assertEqual(item_a.media.data, item_b.media.data, item_a.id)
                test.assertEqual(item_a.media.extra_images, item_b.media.extra_images, item_a.id)
            elif isinstance(item_a.media, Video):
                test.assertEqual(item_a.media, item_b.media, item_a.id)
            elif isinstance(item_a.media, MultiframeImage):
                test.assertEqual(item_a.media.data, item_b.media.data, item_a.id)
        test.assertEqual(len(item_a.annotations), len(item_b.annotations), item_a.id)
        for ann_a in item_a.annotations:
            # We might find few corresponding items, so check them all
            ann_b_matches = [x for x in item_b.annotations if x.type == ann_a.type]
            test.assertFalse(len(ann_b_matches) == 0, "ann id: %s" % ann_a.id)

            ann_b = find(
                ann_b_matches,
                lambda x: _compare_annotations(
                    x,
                    ann_a,
                    ignored_attrs=ignored_attrs,
                    ignore_ann_id=ignore_ann_id,
                    ignore_ann_group=ignore_ann_group,
                ),
            )
            if ann_b is None:
                test.fail("ann %s, candidates %s" % (ann_a, ann_b_matches))
            item_b.annotations.remove(ann_b)  # avoid repeats

    # Check dataset info
    test.assertEqual(expected.infos(), actual.infos())

def compare_datasets_strict(test, expected: IDataset, actual: IDataset, **kwargs):
    # Compares datasets for strong equality

    test.assertEqual(expected.media_type(), actual.media_type())
    test.assertEqual(expected.categories(), actual.categories())

    test.assertListEqual(sorted(expected.subsets()), sorted(actual.subsets()))
    test.assertEqual(len(expected), len(actual))

    for subset_name in expected.subsets():
        e_subset = expected.get_subset(subset_name)
        a_subset = actual.get_subset(subset_name)
        test.assertEqual(len(e_subset), len(a_subset))
        for idx, (item_a, item_b) in enumerate(zip(e_subset, a_subset)):
            test.assertEqual(item_a, item_b, "%s:\n%s\nvs.\n%s\n" % (idx, item_a, item_b))

    # Check dataset info
    test.assertEqual(expected.infos(), actual.infos())

def run_datum(test, *args, expected_code=0):
    from datumaro.cli.__main__ import main

    @contextlib.contextmanager
    def set_no_telemetry():
        from datumaro.util.telemetry_utils import NO_TELEMETRY_KEY

        os.environ[NO_TELEMETRY_KEY] = "1"
        try:
            yield
        finally:
            del os.environ[NO_TELEMETRY_KEY]

    with set_no_telemetry():
        test.assertEqual(expected_code, main(args), str(args))

