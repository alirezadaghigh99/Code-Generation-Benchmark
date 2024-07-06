def export_segmentations(
    sample_collection,
    in_field,
    output_dir,
    rel_dir=None,
    update=True,
    overwrite=False,
    progress=None,
):
    """Exports the segmentations (or heatmaps) stored as in-database arrays in
    the specified field to images on disk.

    Any labels without in-memory arrays are skipped.

    Args:
        sample_collection: a
            :class:`fiftyone.core.collections.SampleCollection`
        in_field: the name of the
            :class:`fiftyone.core.labels.Segmentation` or
            :class:`fiftyone.core.labels.Heatmap` field
        output_dir: the directory in which to write the images
        rel_dir (None): an optional relative directory to strip from each input
            filepath to generate a unique identifier that is joined with
            ``output_dir`` to generate an output path for each image. This
            argument allows for populating nested subdirectories in
            ``output_dir`` that match the shape of the input paths. The path is
            converted to an absolute path (if necessary) via
            :func:`fiftyone.core.storage.normalize_path`
        update (True): whether to delete the arrays from the database
        overwrite (False): whether to delete ``output_dir`` prior to exporting
            if it exists
        progress (None): whether to render a progress bar (True/False), use the
            default value ``fiftyone.config.show_progress_bars`` (None), or a
            progress callback function to invoke instead
    """
    fov.validate_non_grouped_collection(sample_collection)
    fov.validate_collection_label_fields(
        sample_collection, in_field, (fol.Segmentation, fol.Heatmap)
    )

    samples = sample_collection.select_fields(in_field)
    in_field, processing_frames = samples._handle_frame_field(in_field)

    if overwrite:
        etau.delete_dir(output_dir)

    filename_maker = fou.UniqueFilenameMaker(
        output_dir=output_dir, rel_dir=rel_dir, idempotent=False
    )

    for sample in samples.iter_samples(autosave=True, progress=progress):
        if processing_frames:
            images = sample.frames.values()
        else:
            images = [sample]

        for image in images:
            label = image[in_field]
            if label is None:
                continue

            outpath = filename_maker.get_output_path(
                image.filepath, output_ext=".png"
            )

            if isinstance(label, fol.Heatmap):
                if label.map is not None:
                    label.export_map(outpath, update=update)
            else:
                if label.mask is not None:
                    label.export_mask(outpath, update=update)

