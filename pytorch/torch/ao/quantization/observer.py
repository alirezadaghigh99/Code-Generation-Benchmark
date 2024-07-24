class PerChannelMinMaxObserver(UniformQuantizationObserverBase):
    r"""Observer module for computing the quantization parameters based on the
    running per channel min and max values.

    This observer uses the tensor min/max statistics to compute the per channel
    quantization parameters. The module records the running minimum and maximum
    of incoming tensors, and uses this statistic to compute the quantization
    parameters.

    Args:
        ch_axis: Channel axis
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    The quantization parameters are computed the same way as in
    :class:`~torch.ao.quantization.observer.MinMaxObserver`, with the difference
    that the running min/max values are stored per channel.
    Scales and zero points are thus computed per channel as well.

    .. note:: If the running minimum equals to the running maximum, the scales
              and zero_points are set to 1.0 and 0.
    """
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(
        self,
        ch_axis=0,
        dtype=torch.quint8,
        qscheme=torch.per_channel_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        factory_kwargs=None,
        eps=torch.finfo(torch.float32).eps,
        is_dynamic=False,
        **kwargs,
    ) -> None:
        if not is_per_channel(qscheme):
            raise NotImplementedError(
                "PerChannelMinMaxObserver's qscheme only support \
                    torch.per_channel_symmetric, torch.per_channel_affine and torch.per_channel_affine_float_qparams."
            )
        if is_dynamic:
            raise NotImplementedError(
                "PerChannelMinMaxObserver doesn't support dynamic quantization"
            )
        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            factory_kwargs=factory_kwargs,
            eps=eps,
            is_dynamic=is_dynamic,
            **kwargs,
        )
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.ch_axis = ch_axis
        self.register_buffer("min_val", torch.tensor([], **factory_kwargs))
        self.register_buffer("max_val", torch.tensor([], **factory_kwargs))
        if (
            self.qscheme == torch.per_channel_symmetric
            and self.reduce_range
            and self.dtype == torch.quint8
        ):
            raise NotImplementedError(
                "Cannot reduce range for symmetric quantization for quint8"
            )

    def forward(self, x_orig):
        return self._forward(x_orig)

    def _forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        min_val = self.min_val
        max_val = self.max_val
        x_dim = x.size()

        new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
        new_axis_list[self.ch_axis] = 0
        new_axis_list[0] = self.ch_axis
        y = x.permute(new_axis_list)
        # Need to match dtype of min/max because the updates to buffers
        # are done in place and types need to match for comparisons
        y = y.to(self.min_val.dtype)
        y = torch.flatten(y, start_dim=1)
        if min_val.numel() == 0 or max_val.numel() == 0:
            min_val, max_val = torch.aminmax(y, dim=1)
        else:
            min_val_cur, max_val_cur = torch.aminmax(y, dim=1)
            min_val = torch.min(min_val_cur, min_val)
            max_val = torch.max(max_val_cur, max_val)
        self.min_val.resize_(min_val.shape)
        self.max_val.resize_(max_val.shape)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        return self._calculate_qparams(self.min_val, self.max_val)

    def extra_repr(self):
        return f"min_val={self.min_val}, max_val={self.max_val}"

    def _load_from_state_dict(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, torch.Tensor],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ):
        version = local_metadata.get("version", None)
        if version is not None and version < 3:
            local_state = ["min_vals", "max_vals"]
            expected_min_name = "min_vals"
            expected_max_name = "max_vals"
        else:
            local_state = ["min_val", "max_val"]
            expected_min_name = "min_val"
            expected_max_name = "max_val"
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                # Custom handling to allow loading min_val or max_val
                # of size N into uninitialized buffers of size 0. The
                # buffers are resized here, and the values are copied in
                # the default state_dict loading code of the parent.
                if name == expected_min_name:
                    self.min_val.resize_(val.shape)
                elif name == expected_max_name:
                    self.max_val.resize_(val.shape)
                else:
                    warnings.warn(f"Observer load_from_state_dict got unexpected name {name}")
                # For torchscript module we need to update the attributes here since we do not
                # call the `_load_from_state_dict` function defined module.py
                if torch.jit.is_scripting():
                    if name == expected_min_name:
                        self.min_val.copy_(val)
                    elif name == expected_max_name:
                        self.max_val.copy_(val)
                    else:
                        warnings.warn(f"Observer load_from_state_dict got unexpected name {name}")
            elif strict:
                missing_keys.append(key)

        if not torch.jit.is_scripting():
            super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                False,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )

    def _load_from_state_dict_script(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, torch.Tensor],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ):

        self._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @torch.jit.export
    def reset_min_max_vals(self):
        """Resets the min/max values."""
        # This used to be torch.ones but that does not work because
        # JIT compiler can optimize it via common subexpression elimination
        # in which case both min_val and max_val point to the same tensor.
        self.min_val = torch.rand(0, )
        self.max_val = torch.rand(0, )

class HistogramObserver(UniformQuantizationObserverBase):
    r"""
    The module records the running histogram of tensor values along with
    min/max values. ``calculate_qparams`` will calculate scale and zero_point.

    Args:
        bins: Number of bins to use for the histogram
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    The scale and zero point are computed as follows:

    1. Create the histogram of the incoming inputs.
        The histogram is computed continuously, and the ranges per bin change
        with every new tensor observed.
    2. Search the distribution in the histogram for optimal min/max values.
        The search for the min/max values ensures the minimization of the
        quantization error with respect to the floating point model.
    3. Compute the scale and zero point the same way as in the
        :class:`~torch.ao.quantization.MinMaxObserver`
    """
    histogram: torch.Tensor
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(
        self,
        bins: int = 2048,
        dtype: torch.dtype = torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        factory_kwargs=None,
        eps=torch.finfo(torch.float32).eps,
        is_dynamic=False,
        **kwargs,
    ) -> None:
        if not is_per_tensor(qscheme):
            raise NotImplementedError(
                "HistogramObserver's qscheme only support torch.per_tensor_symmetric \
                    and torch.per_tensor_affine."
            )
        if is_dynamic:
            raise NotImplementedError(
                "HistogramObserver doesn't support dynamic quantization"
            )
        # bins: The number of bins used for histogram calculation.
        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            factory_kwargs=factory_kwargs,
            eps=eps,
            is_dynamic=is_dynamic,
            **kwargs,
        )
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.bins = bins
        self.register_buffer("histogram", torch.zeros(self.bins, **factory_kwargs))
        self.register_buffer("min_val", torch.tensor(float("inf"), **factory_kwargs))
        self.register_buffer("max_val", torch.tensor(float("-inf"), **factory_kwargs))
        self.dst_nbins = 2 ** torch.iinfo(self.dtype).bits
        self.upsample_rate = (
            16  # used to reduce quantization errors when upscaling histogram
        )

    def _get_norm(
        self, delta_begin: torch.Tensor, delta_end: torch.Tensor, density: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Compute the norm of the values uniformaly distributed between
        delta_begin and delta_end.
        Currently only L2 norm is supported.

        norm = density * (integral_{begin, end} x^2)
             = density * (end^3 - begin^3) / 3
        """
        norm = (
            delta_end * delta_end * delta_end - delta_begin * delta_begin * delta_begin
        ) / 3
        return density * norm

    def _compute_quantization_error(self, next_start_bin: int, next_end_bin: int):
        r"""
        Compute the quantization error if we use start_bin to end_bin as the
        min and max to do the quantization.
        """
        bin_width = (self.max_val.item() - self.min_val.item()) / self.bins

        dst_bin_width = bin_width * (next_end_bin - next_start_bin + 1) / self.dst_nbins
        if dst_bin_width == 0.0:
            return 0.0

        src_bin = torch.arange(self.bins, device=self.histogram.device)
        # distances from the beginning of first dst_bin to the beginning and
        # end of src_bin
        src_bin_begin = (src_bin - next_start_bin) * bin_width
        src_bin_end = src_bin_begin + bin_width

        # which dst_bins the beginning and end of src_bin belong to?
        dst_bin_of_begin = torch.clamp(
            torch.div(src_bin_begin, dst_bin_width, rounding_mode="floor"),
            0,
            self.dst_nbins - 1,
        )
        dst_bin_of_begin_center = (dst_bin_of_begin + 0.5) * dst_bin_width

        dst_bin_of_end = torch.clamp(
            torch.div(src_bin_end, dst_bin_width, rounding_mode="floor"),
            0,
            self.dst_nbins - 1,
        )
        density = self.histogram / bin_width

        norm = torch.zeros(self.bins, device=self.histogram.device)

        delta_begin = src_bin_begin - dst_bin_of_begin_center
        delta_end = dst_bin_width / 2
        norm += self._get_norm(
            delta_begin,
            torch.ones(self.bins, device=self.histogram.device) * delta_end,
            density,
        )

        norm += (dst_bin_of_end - dst_bin_of_begin - 1) * self._get_norm(
            torch.tensor(-dst_bin_width / 2), torch.tensor(dst_bin_width / 2), density
        )

        dst_bin_of_end_center = dst_bin_of_end * dst_bin_width + dst_bin_width / 2

        delta_begin = -dst_bin_width / 2
        delta_end = src_bin_end - dst_bin_of_end_center
        norm += self._get_norm(torch.tensor(delta_begin), delta_end, density)

        return norm.sum().item()

    def _non_linear_param_search(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Non-linear parameter search.

        An approximation for L2 error minimization for selecting min/max.
        By selecting new min/max, we filter out outliers in input distribution.
        This follows the implementation of NormMinimization::NonlinearQuantizationParamsSearch in
        caffe2/quantization/server/norm_minimization.cc
        """
        assert self.histogram.size()[0] == self.bins, "bins mismatch"
        bin_width = (self.max_val - self.min_val) / self.bins

        # cumulative sum
        total = torch.sum(self.histogram).item()
        cSum = torch.cumsum(self.histogram, dim=0)

        stepsize = 1e-5  # granularity
        alpha = 0.0  # lower bound
        beta = 1.0  # upper bound
        start_bin = 0
        end_bin = self.bins - 1
        norm_min = float("inf")

        while alpha < beta:
            # Find the next step
            next_alpha = alpha + stepsize
            next_beta = beta - stepsize

            # find the left and right bins between the quantile bounds
            l = start_bin
            r = end_bin
            while l < end_bin and cSum[l] < next_alpha * total:
                l = l + 1
            while r > start_bin and cSum[r] > next_beta * total:
                r = r - 1

            # decide the next move
            next_start_bin = start_bin
            next_end_bin = end_bin
            if (l - start_bin) > (end_bin - r):
                # move the start bin
                next_start_bin = l
                alpha = next_alpha
            else:
                # move the end bin
                next_end_bin = r
                beta = next_beta

            if next_start_bin == start_bin and next_end_bin == end_bin:
                continue

            # calculate the quantization error using next_start_bin and next_end_bin
            norm = self._compute_quantization_error(next_start_bin, next_end_bin)

            if norm > norm_min:
                break
            norm_min = norm
            start_bin = next_start_bin
            end_bin = next_end_bin

        new_min = self.min_val + bin_width * start_bin
        new_max = self.min_val + bin_width * (end_bin + 1)
        return new_min, new_max

    def _upscale_histogram(
        self,
        histogram: torch.Tensor,
        orig_min: torch.Tensor,
        orig_max: torch.Tensor,
        update_min: torch.Tensor,
        update_max: torch.Tensor,
    ):
        # this turns the histogram into a more fine-coarsed histogram to reduce
        # bin quantization errors
        histogram = histogram.repeat_interleave(self.upsample_rate) / self.upsample_rate
        bin_size = (orig_max - orig_min) / (self.bins * self.upsample_rate)
        mid_points_histogram = (
            torch.linspace(orig_min, orig_max, self.bins * self.upsample_rate + 1, device=orig_min.device)[
                :-1
            ].to(histogram.device)
            + 0.5 * bin_size
        )
        new_bin_size = (update_min - update_max) / self.bins
        boundaries_new_histogram = torch.linspace(update_min, update_max, self.bins + 1, device=update_min.device).to(
            histogram.device
        )
        # this maps the mid-poits of the histogram to the new histogram's space
        bucket_assignments = (
            torch.bucketize(mid_points_histogram, boundaries_new_histogram) - 1
        )
        # this then maps the histogram mid-points in the new space, weighted by the original histogram's values
        # this is just the old histogram in the new histogram's space

        update_histogram = torch.bincount(
            bucket_assignments, weights=histogram, minlength=self.bins
        )
        return update_histogram

    def _combine_histograms(
        self,
        orig_hist: torch.Tensor,
        orig_min: torch.Tensor,
        orig_max: torch.Tensor,
        update_hist: torch.Tensor,
        update_min: torch.Tensor,
        update_max: torch.Tensor,
    ) -> torch.Tensor:
        # If the new min and max are the same as the current min and max,
        # we can just add the new histogram to the original histogram
        if update_min == orig_min and update_max == orig_max:
            return orig_hist + update_hist

        # If the orig hist only has one value (i.e., the min and max are the same)
        # we can just add it into new histogram
        if orig_min == orig_max:
            bin_value = torch.sum(update_hist)
            transformed_orig_hist = (
                torch.histc(orig_min, bins=self.bins, min=update_min, max=update_max)  # type: ignore[arg-type]
                * bin_value
            )
            return transformed_orig_hist + update_hist

        # We assume the update_hist is already in the target range, we will map the orig_max to it
        assert update_min <= orig_min
        assert update_max >= orig_max

        # Now we need to turn the old_histogram, into the range of the new histogram
        transformed_orig_hist = self._upscale_histogram(
            orig_hist,
            orig_min,
            orig_max,
            update_min,
            update_max,
        )

        return update_hist + transformed_orig_hist

    def reset_histogram(
        self, x: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor
    ) -> None:
        self.min_val.resize_(min_val.shape)
        self.min_val.copy_(min_val)
        self.max_val.resize_(max_val.shape)
        self.max_val.copy_(max_val)
        assert (
            min_val.numel() == 1 and max_val.numel() == 1
        ), "histogram min/max values must be scalar."
        new_histogram = torch.histc(x, self.bins, min=min_val, max=max_val)  # type: ignore[arg-type]
        self.histogram.detach_().resize_(new_histogram.shape)
        self.histogram.copy_(new_histogram)

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:  # pyre-ignore[14]
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()
        x_min, x_max = torch.aminmax(x)
        # want to ignore torch.inf since we don't actually
        # want to make our quantization range infinite
        # and in practice those values will be clamped
        if x_min == -torch.inf or x_max == torch.inf:
            warnings.warn("torch.inf detected in input tensor, ignoring input")
            x = x[x.abs() != torch.inf]
            if x.numel() == 0:
                return x_orig
            x_min, x_max = torch.aminmax(x)

        current_min = self.min_val
        current_max = self.max_val

        is_uninitialized = self.min_val == float("inf") or self.max_val == float("-inf")
        if is_uninitialized:
            self.reset_histogram(x, x_min, x_max)
        else:
            update_min, update_max = x_min, x_max
            new_min = torch.min(current_min, update_min)
            new_max = torch.max(current_max, update_max)

            # TODO: For some reason, this is required for it to pass torchscript test
            # new_min and new_max should already have requires_grad set to False
            new_min, new_max = new_min.detach(), new_max.detach()
            update_histogram = torch.histc(
                x, self.bins, min=new_min, max=new_max  # type: ignore[arg-type]
            ).to(self.histogram.device)
            if new_min == current_min and new_max == current_max:
                combined_histogram = self.histogram + update_histogram
                self.histogram.detach_().resize_(combined_histogram.shape)
                self.histogram.copy_(combined_histogram)
            else:
                combined_histogram = self._combine_histograms(
                    self.histogram,
                    current_min,
                    current_max,
                    update_histogram,
                    new_min,
                    new_max,
                )
                self.histogram.detach_().resize_(combined_histogram.shape)
                self.histogram.copy_(combined_histogram)
                self.min_val.detach_().resize_(new_min.shape)
                self.min_val.copy_(new_min)
                self.max_val.detach_().resize_(new_max.shape)
                self.max_val.copy_(new_max)

        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        is_uninitialized = self.min_val == float("inf") and self.max_val == float(
            "-inf"
        )
        if is_uninitialized:
            warnings.warn(
                "must run observer before calling calculate_qparams.\
                                    Returning default scale and zero point "
            )
            return torch.tensor([1.0], device=self.min_val.device.type), torch.tensor(
                [0], device=self.min_val.device.type
            )
        assert self.bins == len(self.histogram), (
            "The number of bins in histogram should be equal to the number of bins "
            "supplied while making this observer"
        )

        new_min, new_max = self._non_linear_param_search()

        return self._calculate_qparams(new_min, new_max)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "min_val"] = self.min_val
        destination[prefix + "max_val"] = self.max_val

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 3:
            # if min_val and max_val are not initialized, update their shape
            # to account for the differences between v2 and v3
            min_val_name, max_val_name = prefix + "min_val", prefix + "max_val"
            if min_val_name in state_dict:
                if state_dict[min_val_name].shape == torch.Size([0]):
                    state_dict[min_val_name] = torch.tensor(float("inf"))
            if max_val_name in state_dict:
                if state_dict[max_val_name].shape == torch.Size([0]):
                    state_dict[max_val_name] = torch.tensor(float("-inf"))

        local_state = ["min_val", "max_val"]
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                setattr(self, name, val)
            elif strict:
                missing_keys.append(key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def extra_repr(self):
        return f"min_val={self.min_val}, max_val={self.max_val}"

class MovingAverageMinMaxObserver(MinMaxObserver):
    r"""Observer module for computing the quantization parameters based on the
    moving average of the min and max values.

    This observer computes the quantization parameters based on the moving
    averages of minimums and maximums of the incoming tensors. The module
    records the average minimum and maximum of incoming tensors, and uses this
    statistic to compute the quantization parameters.

    Args:
        averaging_constant: Averaging constant for min/max.
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    The moving average min/max is computed as follows

    .. math::

        \begin{array}{ll}
                x_\text{min} = \begin{cases}
                    \min(X) & \text{if~}x_\text{min} = \text{None} \\
                    (1 - c) x_\text{min} + c \min(X) & \text{otherwise}
                \end{cases}\\
                x_\text{max} = \begin{cases}
                    \max(X) & \text{if~}x_\text{max} = \text{None} \\
                    (1 - c) x_\text{max} + c \max(X) & \text{otherwise}
                \end{cases}\\
        \end{array}

    where :math:`x_\text{min/max}` is the running average min/max, :math:`X` is
    is the incoming tensor, and :math:`c` is the ``averaging_constant``.

    The scale and zero point are then computed as in
    :class:`~torch.ao.quantization.observer.MinMaxObserver`.

    .. note:: Only works with ``torch.per_tensor_affine`` quantization scheme.

    .. note:: If the running minimum equals to the running maximum, the scale
              and zero_point are set to 1.0 and 0.
    """

    def __init__(
        self,
        averaging_constant=0.01,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        eps=torch.finfo(torch.float32).eps,
        is_dynamic=False,
        **kwargs
    ) -> None:
        if not is_per_tensor(qscheme):
            raise NotImplementedError(
                f"MovingAverageMinMaxObserver's qscheme only support \
                torch.per_tensor_symmetric and torch.per_tensor_affine. \
                but got: {qscheme}"
            )
        self.averaging_constant = averaging_constant
        if is_dynamic and self.averaging_constant != 1:
            raise NotImplementedError(
                "MovingAverageMinMaxObserver doesn't support dynamic quantization for "
                f"averaging constant of {self.averaging_constant}"
            )
        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            eps=eps,
            is_dynamic=is_dynamic,
            **kwargs
        )

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        min_val = self.min_val
        max_val = self.max_val
        if min_val == float("inf") and max_val == float("-inf"):
            min_val, max_val = torch.aminmax(x)
        else:
            min_val_cur, max_val_cur = torch.aminmax(x)
            min_val = min_val + self.averaging_constant * (min_val_cur - min_val)
            max_val = max_val + self.averaging_constant * (max_val_cur - max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

