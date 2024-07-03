    def _pad_strides(in_strides, size, dtype):
        """
        The padding does not change stride order but makes sure all strides larger
        than the threshold are multiple of align.
        """
        align = get_align_for_dtype(dtype)
        if len(in_strides) == 0:
            return in_strides

        if not config.pad_channels_last and Layout.is_channels_last_contiguous(
            size, in_strides
        ):
            return in_strides

        current_fx_node = V.get_current_node()
        if hasattr(current_fx_node, "meta") and current_fx_node.meta.get(
            "dislike_padding", False
        ):
            return in_strides

        # get_stride_order does not work with dynamic shape. Also we can not
        # statically decide if a padding is needed or how much padding we should
        # do for dynamic shape.
        #
        # Skip padding the strides for dynamic shape for now.
        if not all(
            isinstance(s, (int, sympy.Integer))
            for s in itertools.chain(in_strides, size)
        ):
            return in_strides

        stride_order = get_stride_order(in_strides)
        fill_order = stride_order2fill_order(stride_order)

        new_strides = [0 for _ in range(len(in_strides))]
        # since we pad when the layout is flexible, we can decide the
        # smallest stride to be 1.
        new_strides[fill_order[0]] = 1

        # Don't align a too small stride since that causes too much memory increase.
        # Pad too small stride may also cause perf loss. We may result in many tiny data blocks
        # with gaps in between. That causes less coalesced GPU memory access!
        #
        # Initially we pick 320 as the threshold since for alignement=16,
        # that results in at most 5% memory cost.
        #
        # But later on we raise the threshold to 1024 to avoid interfere with persistent reduction.
        # Let's say an inner reduction has a row size 513. Inductor will generate
        # persistent reduction code.
        # If we do padding, the strides are not contiguous any more. Inductor
        # uses a much smaller threshold for persistent reduction in this case and
        # generates potentially worse non-persistent reduction code.
        #
        # This change turns HF AllenaiLongformerBase amp training from a loss of 1.09x to a win of 1.05x.
        # (baseline: 71.09ms, padding w/o this change: 77.38ms, padding with this change: 67.77ms)
        align_stride_threshold = 1024
        padded = False
        for rank, idx in enumerate(fill_order[1:], start=1):
            prev_idx = fill_order[rank - 1]
            stride = new_strides[prev_idx] * size[prev_idx]

            if stride > align_stride_threshold and stride % align != 0:
                stride = ceildiv(stride, align) * align
                padded = True
            new_strides[idx] = stride

        if not padded:
            # Consider a tensor with shape [256, 1, 5, 5]
            # Avoid strides like [25, 5, 5, 1] being padded to equivalent strides
            # [25, 25, 5, 1].
            return in_strides

        metrics.num_comprehensive_padding += 1
        return new_strides