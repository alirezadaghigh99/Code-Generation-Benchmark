class profile:
    """Context manager that manages autograd profiler state and holds a summary of results.

    Under the hood it just records events of functions being executed in C++ and
    exposes those events to Python. You can wrap any code into it and it will
    only report runtime of PyTorch functions.
    Note: profiler is thread local and is automatically propagated into the async tasks

    Args:
        enabled (bool, optional): Setting this to False makes this context manager a no-op.

        use_cuda (bool, optional): Enables timing of CUDA events as well
            using the cudaEvent API. (will be deprecated)

        use_device (str, optional): Enables timing of device events.
            Adds approximately 4us of overhead to each tensor operation when use cuda.
            The valid devices options are 'cuda', 'xpu', 'mtia' and 'privateuseone'.

        record_shapes (bool, optional): If shapes recording is set, information
            about input dimensions will be collected. This allows one to see which
            dimensions have been used under the hood and further group by them
            using prof.key_averages(group_by_input_shape=True). Please note that
            shape recording might skew your profiling data. It is recommended to
            use separate runs with and without shape recording to validate the timing.
            Most likely the skew will be negligible for bottom most events (in a case
            of nested function calls). But for higher level functions the total
            self cpu time might be artificially increased because of the shape
            collection.

        with_flops (bool, optional): If with_flops is set, the profiler will estimate
            the FLOPs (floating point operations) value using the operator's input shape.
            This allows one to estimate the hardware performance. Currently,
            this option only works for the matrix multiplication and 2D convolution operators.

        profile_memory (bool, optional): track tensor memory allocation/deallocation.

        with_stack (bool, optional): record source information (file and line number) for the ops.

        with_modules (bool): record module hierarchy (including function names)
            corresponding to the callstack of the op. e.g. If module A's forward call's
            module B's forward which contains an aten::add op,
            then aten::add's module hierarchy is A.B
            Note that this support exist, at the moment, only for TorchScript models
            and not eager mode models.

        use_kineto (bool, optional): experimental, enable profiling with Kineto profiler.

        use_cpu (bool, optional): profile CPU events; setting to ``False`` requires
            ``use_kineto=True`` and can be used to lower the overhead for GPU-only profiling.

        experimental_config (_ExperimentalConfig) : A set of experimental options
            used by profiler libraries like Kineto. Note, backward compatibility is not guaranteed.


    .. warning:
        Enabling memory profiling or source attribution incurs additional profiler
        overhead

    .. warning:
        This context managers should not be called recursively, i.e. no nested
        instances are allowed

    .. warning:
        Due to some CUDA multiprocessing limitations (multiprocessing-cuda-note_),
        one cannot use the profiler with ``use_device = 'cuda'`` to benchmark
        DataLoaders with ``num_workers > 0``. If you wish to benchmark data loading,
        please use ``use_device = None`` or ``num_workers = 0``.

    Example:
        >>> # xdoctest: +SKIP
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD_PROFILER)
        >>> x = torch.randn((1, 1), requires_grad=True)
        >>> with torch.autograd.profiler.profile() as prof:
        >>>     for _ in range(100):  # any normal python code, really!
        >>>         y = x ** 2
        >>>         y.backward()
        >>> # NOTE: some columns were removed for brevity
        >>> print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        -----------------------------------  ---------------  ---------------  ---------------
        Name                                 Self CPU total   CPU time avg     Number of Calls
        -----------------------------------  ---------------  ---------------  ---------------
        mul                                  32.048ms         32.048ms         200
        pow                                  27.041ms         27.041ms         200
        PowBackward0                         9.727ms          55.483ms         100
        torch::autograd::AccumulateGrad      9.148ms          9.148ms          100
        torch::autograd::GraphRoot           691.816us        691.816us        100
        -----------------------------------  ---------------  ---------------  ---------------

    """

    def __init__(
        self,
        enabled=True,
        *,
        use_cuda=False,  # Deprecated
        use_device=None,
        record_shapes=False,
        with_flops=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        use_kineto=False,
        use_cpu=True,
        experimental_config=None,
    ):
        self.enabled: bool = enabled
        if not self.enabled:
            return
        self.use_cuda = use_cuda
        if self.use_cuda:
            warn(
                "The attribute `use_cuda` will be deprecated soon, "
                "please use ``use_device = 'cuda'`` instead.",
                FutureWarning,
                stacklevel=2,
            )
            self.use_device: Optional[str] = "cuda"
        else:
            self.use_device = use_device
        self.function_events: Optional[EventList] = None
        self.entered = False
        self.record_shapes = record_shapes
        self.with_flops = with_flops
        self.record_shapes |= self.with_flops
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_modules = with_modules
        self.use_cpu = use_cpu
        if experimental_config is None:
            experimental_config = _ExperimentalConfig()
        self.experimental_config = experimental_config
        self.kineto_results: Optional[_ProfilerResult] = None
        self.profiling_start_time_ns = 0
        self.profiling_end_time_ns = 0
        self._stats = _ProfilerStats()

        if not self.use_cpu:
            assert (
                use_kineto
            ), "Device-only events supported only with Kineto (use_kineto=True)"

        if self.use_device is not None:
            VALID_DEVICE_OPTIONS = ["cuda", "xpu", "mtia"]
            if _get_privateuse1_backend_name() != "privateuseone":
                VALID_DEVICE_OPTIONS.append(_get_privateuse1_backend_name())
            if self.use_device not in VALID_DEVICE_OPTIONS:
                warn(f"The {self.use_device} is not a valid device option.")
                self.use_device = None

            if self.use_device == "cuda" and not torch.cuda.is_available():
                warn("CUDA is not available, disabling CUDA profiling")
                self.use_cuda = False
                self.use_device = None

            if self.use_device == "xpu" and not torch.xpu.is_available():
                warn("XPU is not available, disabling XPU profiling")
                self.use_device = None

        self.kineto_activities = set()
        if self.use_cpu:
            self.kineto_activities.add(ProfilerActivity.CPU)

        self.profiler_kind = ProfilerState.KINETO
        if self.use_device == "cuda":
            if not use_kineto or ProfilerActivity.CUDA not in _supported_activities():
                assert self.use_cpu, "Legacy CUDA profiling requires use_cpu=True"
                self.profiler_kind = ProfilerState.KINETO_GPU_FALLBACK
            else:
                self.kineto_activities.add(ProfilerActivity.CUDA)
        elif self.use_device == "xpu":
            assert (
                use_kineto and ProfilerActivity.XPU in _supported_activities()
            ), "Legacy XPU profiling is not supported. Requires use_kineto=True on XPU devices."
            self.kineto_activities.add(ProfilerActivity.XPU)
        elif self.use_device == "mtia":
            assert (
                use_kineto and ProfilerActivity.MTIA in _supported_activities()
            ), "Legacy MTIA profiling is not supported. Requires use_kineto=True on MTIA devices."
            self.kineto_activities.add(ProfilerActivity.MTIA)
        elif self.use_device is not None and self.use_device != "privateuseone":
            if (
                not use_kineto
                or ProfilerActivity.PrivateUse1 not in _supported_activities()
            ):
                assert (
                    self.use_cpu
                ), "Legacy custombackend profiling requires use_cpu=True"
                self.profiler_kind = ProfilerState.KINETO_PRIVATEUSE1_FALLBACK
            else:
                self.kineto_activities.add(ProfilerActivity.PrivateUse1)

        assert (
            len(self.kineto_activities) > 0
        ), "No activities specified for the profiler"

    def config(self):
        return ProfilerConfig(
            self.profiler_kind,
            self.record_shapes,
            self.profile_memory,
            self.with_stack,
            self.with_flops,
            self.with_modules,
            self.experimental_config,
        )

    def __enter__(self):
        if not self.enabled:
            return
        if self.entered:
            raise RuntimeError("Profiler context manager is not reentrant")
        self._prepare_trace()
        self._start_trace()
        return self

    def _prepare_trace(self):
        self.entered = True
        t0 = perf_counter_ns()
        _prepare_profiler(self.config(), self.kineto_activities)
        t1 = perf_counter_ns()
        self._stats.profiler_prepare_call_duration_us = int((t1 - t0) / 1000)

    def _start_trace(self):
        self.entered = True
        _run_on_profiler_start()
        t0 = perf_counter_ns()
        _enable_profiler(self.config(), self.kineto_activities)
        t1 = perf_counter_ns()
        self._stats.profiler_enable_call_duration_us = int((t1 - t0) / 1000)
        self.profiling_start_time_ns = t1

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        if self.use_device and hasattr(torch, self.use_device):
            device_module = getattr(torch, self.use_device)
            if hasattr(device_module, "synchronize"):
                device_module.synchronize()

        old_function_events: Optional[EventList] = None
        if self.function_events:
            old_function_events = self.function_events

        t0 = perf_counter_ns()

        # TODO we are overwriting previous kineto results here
        # Should combine previous results with the new results otherwise only
        # the last "repeat" will be recorded in the trace
        self.kineto_results = _disable_profiler()
        t1 = perf_counter_ns()
        self._stats.profiler_disable_call_duration_us = int((t1 - t0) / 1000)
        self.profiling_end_time_ns = t0

        _run_on_profiler_stop()
        t0 = perf_counter_ns()
        parsed_results = self._parse_kineto_results(self.kineto_results)
        t1 = perf_counter_ns()
        self._stats.parse_kineto_call_duration_us = int((t1 - t0) / 1000)

        self.function_events = EventList(
            parsed_results,
            use_device=self.use_device,
            profile_memory=self.profile_memory,
            with_flops=self.with_flops,
        )
        t0 = perf_counter_ns()
        self.function_events._build_tree()
        t1 = perf_counter_ns()
        self._stats.function_events_build_tree_call_duration_us = int((t1 - t0) / 1000)

        self._stats.number_of_events = len(self.function_events)
        self._stats.profiling_window_duration_sec = (
            (self.profiling_end_time_ns - self.profiling_start_time_ns) * 1.0 / 1e9
        )
        if old_function_events:
            for evt in old_function_events:
                self.function_events.append(evt)
        return False

    def __repr__(self):
        if self.function_events is None:
            return "<unfinished torch.autograd.profile>"
        return repr(self.function_events)

    def __str__(self):
        if self.function_events is None:
            return "<unfinished torch.autograd.profile>"
        return str(self.function_events)

    def _check_finish(self):
        if self.function_events is None:
            raise RuntimeError("Profiler didn't finish running")

    def table(
        self,
        sort_by=None,
        row_limit=100,
        max_src_column_width=75,
        max_name_column_width=55,
        max_shapes_column_width=80,
        header=None,
        top_level_events_only=False,
    ):
        self._check_finish()
        assert self.function_events is not None
        return self.function_events.table(
            sort_by=sort_by,
            row_limit=row_limit,
            max_src_column_width=max_src_column_width,
            max_name_column_width=max_name_column_width,
            max_shapes_column_width=max_shapes_column_width,
            header=header,
            top_level_events_only=top_level_events_only,
        )

    table.__doc__ = EventList.table.__doc__

    def export_chrome_trace(self, path):
        """
        Exports the collected trace in Chrome JSON format. If kineto is enabled, only
        last cycle in schedule is exported.
        """
        self._check_finish()
        if kineto_available():
            self.kineto_results.save(path)  # type: ignore[union-attr]
        else:
            return self.function_events.export_chrome_trace(path)  # type: ignore[union-attr]

    export_chrome_trace.__doc__ = EventList.export_chrome_trace.__doc__

    def export_stacks(self, path: str, metric: str = "self_cpu_time_total"):
        self._check_finish()
        assert self.function_events is not None, "Expected profiling results"
        assert self.with_stack, "export_stacks() requires with_stack=True"
        return self.function_events.export_stacks(path, metric)

    def key_averages(self, group_by_input_shape=False, group_by_stack_n=0):
        self._check_finish()
        assert self.function_events is not None, "Expected profiling results"
        return self.function_events.key_averages(group_by_input_shape, group_by_stack_n)

    key_averages.__doc__ = EventList.key_averages.__doc__

    def total_average(self):
        self._check_finish()
        assert self.function_events is not None, "Expected profiling results"
        return self.function_events.total_average()

    total_average.__doc__ = EventList.total_average.__doc__

    @property
    def self_cpu_time_total(self):
        """Returns total time spent on CPU.

        The total time is a sum of all self times across all the events.
        """
        self._check_finish()
        assert self.function_events is not None
        return self.function_events.self_cpu_time_total

    def _parse_kineto_results(self, result: _ProfilerResult):
        # result.events() has most of the events - PyTorch op-level and device-level events

        trace_start_ns = result.trace_start_ns()
        mem_records = [
            [evt, False] for evt in result.events() if evt.name() == MEMORY_EVENT_NAME
        ]
        oom_records = [
            evt for evt in result.events() if evt.name() == OUT_OF_MEMORY_EVENT_NAME
        ]
        mem_records_acc = MemRecordsAcc(mem_records)

        def _cpu_memory_usage(mem_record):
            return (
                mem_record.nbytes()
                if mem_record.device_type()
                in [DeviceType.CPU, DeviceType.MKLDNN, DeviceType.IDEEP]
                else 0
            )

        def _device_memory_usage(mem_record):
            return (
                mem_record.nbytes()
                if mem_record.device_type()
                in [DeviceType.CUDA, DeviceType.PrivateUse1, DeviceType.HIP]
                else 0
            )

        # Create and return FunctionEvent list, which contains all function events
        # Here 2 function events are created:
        # all_function_events contains all events associated with each kineto event from result
        all_function_events = []
        # frontend_function_events contains the events in aten or torch frontend level,
        # whose correlation id is 0
        frontend_function_events = []
        device_corr_map: Dict[int, List[FunctionEvent]] = {}
        max_evt_id = 0
        for kineto_event in result.events():
            if _filter_name(kineto_event.name()):
                continue
            rel_start_ns = kineto_event.start_ns() - trace_start_ns
            rel_end_ns = kineto_event.end_ns() - trace_start_ns
            abs_end_ns = kineto_event.end_ns()

            cpu_memory_usage = 0
            device_memory_usage = 0
            if kineto_event.device_type() == DeviceType.CPU:
                # find the corresponding memory allocation events
                for mem_record in mem_records_acc.in_interval(
                    kineto_event.start_ns() / 1000, abs_end_ns / 1000
                ):
                    cpu_memory_usage += _cpu_memory_usage(mem_record[0])
                    device_memory_usage += _device_memory_usage(mem_record[0])
                    mem_record[1] = True

            is_async = kineto_event.is_async() or (
                kineto_event.start_thread_id() != kineto_event.end_thread_id()
            )

            fe = FunctionEvent(
                id=kineto_event.correlation_id(),
                name=_rewrite_name(name=kineto_event.name(), with_wildcard=True),
                trace_name=_rewrite_name(name=kineto_event.name(), with_wildcard=False),
                thread=kineto_event.start_thread_id(),
                start_us=rel_start_ns / 1000,
                end_us=rel_end_ns / 1000,
                fwd_thread=kineto_event.fwd_thread_id(),
                input_shapes=kineto_event.shapes(),
                concrete_inputs=kineto_event.concrete_inputs(),
                stack=[
                    entry
                    for entry in kineto_event.stack()
                    if _filter_stack_entry(entry)
                ],
                scope=kineto_event.scope(),
                use_device=self.use_device,
                cpu_memory_usage=cpu_memory_usage,
                device_memory_usage=device_memory_usage,
                is_async=is_async,
                sequence_nr=kineto_event.sequence_nr(),
                device_type=kineto_event.device_type(),
                device_index=kineto_event.device_index(),
                device_resource_id=kineto_event.device_resource_id(),
                flops=kineto_event.flops(),
            )
            max_evt_id = max(max_evt_id, fe.id)
            if fe.device_type == DeviceType.CPU and not fe.is_async:
                if self.use_device == "privateuseone":
                    privateuse1_time = kineto_event.privateuse1_elapsed_us()
                    if privateuse1_time > 0:
                        fe.append_kernel(fe.name, fe.device_index, privateuse1_time)
                        fe.is_legacy = True
                elif self.use_device == "cuda":
                    # Check if we have CUDA time as a fallback
                    cuda_time = kineto_event.cuda_elapsed_us()
                    if cuda_time > 0:
                        fe.append_kernel(fe.name, fe.device_index, cuda_time)
                        fe.is_legacy = True
            all_function_events.append(fe)
            corr_id = kineto_event.linked_correlation_id()
            if corr_id > 0:
                if corr_id not in device_corr_map:
                    device_corr_map[corr_id] = []
                device_corr_map[corr_id].append(fe)
            elif corr_id == 0:
                frontend_function_events.append(fe)
            else:
                raise RuntimeError(
                    f"Got negative correlation id {corr_id} in profiler post processing"
                )

        # associate device kernels and device runtime (CPU) with CPU events
        for fe in frontend_function_events:
            if (
                fe.device_type == DeviceType.CPU
                and not fe.is_async
                and fe.id in device_corr_map
            ):
                for f_evt in device_corr_map[fe.id]:
                    if (
                        f_evt.device_type == DeviceType.CUDA
                        or f_evt.device_type == DeviceType.PrivateUse1
                    ):
                        fe.append_kernel(
                            f_evt.name,
                            f_evt.device_index,
                            f_evt.time_range.end - f_evt.time_range.start,
                        )
                    elif f_evt.device_type == DeviceType.CPU:
                        # make sure that 'thread' of a CPU Kineto (e.g. Device Runtime) event is associated
                        # with the 'thread' of the corresponding linked PyTorch event to properly track
                        # parents and children
                        f_evt.thread = fe.thread

        def createFunctionEventForMemoryEvents(evt):
            rel_start_ns = evt.start_ns() - trace_start_ns
            fe = FunctionEvent(
                id=max_evt_id,
                name=evt.name(),
                trace_name=None,  # not outputting in the trace
                thread=evt.start_thread_id(),
                start_us=rel_start_ns / 1000,
                end_us=rel_start_ns / 1000,  # no duration
                fwd_thread=evt.start_thread_id(),
                input_shapes=[],
                stack=[],
                scope=0,  # RecordScope::FUNCTION
                use_device=self.use_device,
                cpu_memory_usage=_cpu_memory_usage(evt),
                device_memory_usage=_device_memory_usage(evt),
                is_async=False,
                sequence_nr=-1,
                device_type=DeviceType.CPU,
                device_index=0,
            )
            return fe

        # output top-level memory events
        for mem_record in mem_records:
            if not mem_record[1]:
                max_evt_id += 1
                fe = createFunctionEventForMemoryEvents(mem_record[0])
                all_function_events.append(fe)

        for oom_record in oom_records:
            max_evt_id += 1
            fe = createFunctionEventForMemoryEvents(oom_record)
            all_function_events.append(fe)

        all_function_events.sort(
            key=lambda evt: [evt.time_range.start, -evt.time_range.end]
        )
        return all_function_events

class record_function(_ContextDecorator):
    """Context manager/function decorator that adds a label to a code block/function when running autograd profiler.

    It is useful when tracing the code profile.

    Args:
        name (str): Label assigned to the block of code.
        node_id (int): ID of node, for distributed profiling. Unset in
        non-distributed cases.

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD_PROFILER)
        >>> x = torch.randn((1, 1), requires_grad=True)
        >>> with torch.autograd.profiler.profile() as prof:
        ...     y = x ** 2
        ...     with torch.autograd.profiler.record_function("label-z"): # label the block
        ...         z = y ** 3
        ...     y.backward()
        ...
        >>> # xdoctest: +IGNORE_WANT
        >>> # NOTE: some columns were removed for brevity
        >>> print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        -----------------------------------  ---------------  ---------------  ---------------
        Name                                 Self CPU total %  CPU time avg     Number of Calls
        -----------------------------------  ---------------  ---------------  ---------------
        pow                                  60.77%           47.470us         3
        mul                                  21.73%           25.465us         2
        PowBackward0                         12.03%           121.891us        1
        torch::autograd::AccumulateGrad      2.70%            6.324us          1
        label-z                              2.13%            12.421us         1
        torch::autograd::GraphRoot           0.64%            1.503us          1
        -----------------------------------  ---------------  ---------------  ---------------
        Self CPU time total: 234.344us
        CUDA time total: 0.000us

    """

    def __init__(self, name: str, args: Optional[str] = None):
        self.name: str = name
        self.args: Optional[str] = args
        # Whether or not we should run record function's end callbacks when exiting.
        self.run_callbacks_on_exit: bool = True
        # TODO: TorchScript ignores standard type annotation here
        # self.record: Optional["torch.classes.profiler._RecordFunction"] = None
        self.record = torch.jit.annotate(
            Optional["torch.classes.profiler._RecordFunction"], None
        )

    def __enter__(self):
        self.record = torch.ops.profiler._record_function_enter_new(
            self.name, self.args
        )
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any):
        if not self.run_callbacks_on_exit:
            return

        # Local variable is needed by TorchScript to refine Optional[T] to T
        record = self.record
        assert record is not None

        # TODO: Too slow with __torch_function__ handling enabled
        # See https://github.com/pytorch/pytorch/issues/76410
        if not torch.jit.is_scripting():
            with torch._C.DisableTorchFunctionSubclass():
                torch.ops.profiler._record_function_exit._RecordFunction(record)
        else:
            torch.ops.profiler._record_function_exit(record)

    def _call_end_callbacks_on_future(self, fut: Future[Any]) -> Future[Any]:
        """Use for profiling async calls that return a future.

        Calling this function will extend recording beyond this scope, until the future is
        satisfied. It is useful for profiling the end to end time of asynchronous calls.
        This function should only be called once to attach the callback onto the future, and
        will throw if called multiple times.

        Args:
            fut: (torch._C.Future): future for which to schedule
            callback for.

        Returns:
            A future that completes with the value of the passed in future when
            the profiling callbacks have ran.

        """
        # Throw if we have already attached a callback onto the future.
        if not self.run_callbacks_on_exit:
            raise RuntimeError("_call_end_callbacks_on_future can only be called once.")

        # We are scheduling to run this RecordFunction's end callbacks when the
        # passed in future completes, so don't run end callbacks on exit.
        self.run_callbacks_on_exit = False

        # Local variable is needed by TorchScript to refine Optional[T] to T
        record = self.record
        assert record is not None

        # TODO: Too slow with __torch_function__ handling enabled
        # See https://github.com/pytorch/pytorch/issues/76410
        if not torch.jit.is_scripting():
            with torch._C.DisableTorchFunctionSubclass():
                profiled_future = (
                    torch.ops.profiler._call_end_callbacks_on_jit_fut._RecordFunction(
                        record, fut
                    )
                )
        else:
            profiled_future = torch.ops.profiler._call_end_callbacks_on_jit_fut(
                record, fut
            )
        return profiled_future

