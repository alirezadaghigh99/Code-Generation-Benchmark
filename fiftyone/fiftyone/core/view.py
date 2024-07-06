def make_optimized_select_view(
    sample_collection,
    sample_ids,
    ordered=False,
    groups=False,
    flatten=False,
):
    """Returns a view that selects the provided sample IDs that is optimized
    to reduce the document list as early as possible in the pipeline.

    .. warning::

        This method **deletes** any other view stages that reorder/select
        documents, so the returned view may not respect the order of the
        documents in the input collection.

    Args:
        sample_collection:  a
            :class:`fiftyone.core.collections.SampleCollection`
        sample_ids: a sample ID or iterable of sample IDs to select
        ordered (False): whether to sort the samples in the returned view to
            match the order of the provided IDs
        groups (False): whether the IDs are group IDs, not sample IDs
        flatten (False): whether to flatten group datasets before selecting
            sample ids

    Returns:
        a :class:`DatasetView`
    """
    in_view = sample_collection.view()
    stages = in_view._stages

    if any(isinstance(stage, fost.Mongo) for stage in stages):
        # We have no way of knowing what a `Mongo()` stage might do, so we must
        # run the entire view's aggregation first and then select the samples
        # of interest at the end
        view = in_view
        stages = []
    else:
        view = in_view._base_view

    if groups:
        view = view.select_groups(sample_ids, ordered=ordered)
    else:
        if view.media_type == fom.GROUP and view.group_slices and flatten:
            view = view.select_group_slices(_allow_mixed=True)
        else:
            for stage in stages:
                if type(stage) in fost._STAGES_THAT_SELECT_FIRST:
                    view = view._add_view_stage(stage, validate=False)

        view = view.select(sample_ids, ordered=ordered)

    #
    # Selecting the samples of interest first can be significantly faster than
    # running the entire aggregation and then selecting them.
    #
    # However, in order to do that, we must omit any `Skip()` stages, which
    # depend on the number of documents in the pipeline.
    #
    # In addition, we take the liberty of omitting other stages that are known
    # to only select/reorder documents.
    #
    # @note this is brittle because if any new stages like `Skip()` are added
    # that could affect our ability to select the samples of interest first,
    # we'll need to account for that here...
    #

    for stage in stages:
        if type(stage) not in fost._STAGES_THAT_SELECT_OR_REORDER:
            view = view._add_view_stage(stage, validate=False)

    return view

