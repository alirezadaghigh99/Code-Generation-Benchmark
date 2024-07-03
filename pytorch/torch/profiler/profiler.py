class ExecutionTraceObserver(_ITraceObserver):
    """Execution Trace Observer

    Each process can have a single ExecutionTraceObserver instance. The observer
    can be added to record function callbacks via calling register_callback()
    explicitly. Without calling unregister_callback(), repeated calls to
    register_callback() will not add additional observers to record function
    callbacks. Once an ExecutionTraceObserver is created, the start() and stop()
    methods control when the event data is recorded.

    Deleting or calling unregister_callback() will remove the observer from the
    record function callbacks, finalize the output file, and will stop
    incurring any overheads.
    """

    def __init__(self):
        """
        Initializes the default states.
        """
        self._registered = False
        self._execution_trace_running = False

    def __del__(self):
        """
        Calls unregister_callback() to make sure to finalize outputs.
        """
        self.unregister_callback()

    def register_callback(self, output_file_path: str) -> Self:
        """
        Adds ET observer to record function callbacks. The data will be
        written to output_file_path.
        """
        if not self._registered:
            self._output_file_path = output_file_path
            self._registered = _add_execution_trace_observer(output_file_path)
        return self

    def unregister_callback(self):
        """
        Removes ET observer from record function callbacks.
        """

        def _save_triton_kernels():
            # Save the kernel paths for the generated kernels
            from torch._inductor.codecache import PyCodeCache as PyCodeCache

            kernel_files = [
                v.__file__
                for v in PyCodeCache.cache.values()
                if getattr(v, "__file__", None) is not None
            ]
            work_dir, file_name = os.path.split(self._output_file_path)
            resource_dir = os.path.join(
                work_dir, os.path.splitext(file_name)[0] + "_resources"
            )
            if not os.path.exists(resource_dir):
                os.mkdir(resource_dir)

            for kernel_file in kernel_files:
                if kernel_file is None:
                    continue
                path, name = os.path.split(kernel_file)
                dst = os.path.join(resource_dir, name)
                shutil.copyfile(kernel_file, dst)

        if self._registered:
            self.stop()
            try:
                _save_triton_kernels()
            except Exception as e:
                warn(f"Execution trace failed to save kernels: {e}")
            _remove_execution_trace_observer()
            self._registered = False

    @property
    def is_registered(self):
        """
        Returns True if the execution trace observer is registered, otherwise False.
        """
        return self._registered

    def is_running(self):
        """
        Returns True if the observer is running, otherwise False.
        """
        return self._execution_trace_running

    def start(self):
        """
        Starts to capture.
        """
        if self._registered and not self._execution_trace_running:
            _enable_execution_trace_observer()
            self._execution_trace_running = True
            self._record_pg_config()

    def stop(self):
        """
        Stops to capture.
        """
        if self._execution_trace_running:
            _disable_execution_trace_observer()
            self._execution_trace_running = False

    def cleanup(self):
        """
        Calls unregister_callback() to make sure to finalize outputs.
        """
        self.unregister_callback()

    def get_output_file_path(self) -> str:
        """
        Returns the output file name.
        """
        if self.is_registered:
            return self._output_file_path
        else:
            raise RuntimeError(
                "A callback to the ET profiler needs to be registered "
                "first before getting the output file path"
            )

    def _record_pg_config(self) -> None:
        # Records the PG config info to the trace as node:
        #  ## process_group:init ##
        if (
            self.is_registered
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            pg_config_info = torch.distributed.distributed_c10d._world.pg_config_info
            torch.autograd._record_function_with_args_enter(
                "## process_group:init ##", json.dumps(pg_config_info)
            )