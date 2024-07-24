class CachingAutotuner(KernelInterface):
    """
    Simplified version of Triton autotuner that has no invalidation
    key and caches the best config to disk to improve cold start times.
    Unlike the main triton Autotuner, this version can precompile all
    configs, and does not rely on the Triton JIT.
    """

    def __init__(
        self,
        fn,
        triton_meta,  # passed directly to triton
        configs,
        save_cache_hook,
        mutated_arg_names,
        heuristic_type,
        size_hints=None,
        inductor_meta=None,  # metadata not relevant to triton
        custom_kernel=False,  # whether the kernel is inductor-generated or custom
        filename: Optional[str] = None,
    ):
        super().__init__()

        assert len(configs) > 0, "Non-empty TritonConfig list required for compiling"
        self.fn = fn
        self.device_props: DeviceProperties = triton_meta["device"]
        self.triton_meta = {
            **triton_meta,
            "device": self.device_props.index,
            "device_type": self.device_props.type,
        }
        self.inductor_meta = {} if inductor_meta is None else inductor_meta
        self.save_cache_hook = save_cache_hook
        self.mutated_arg_names = mutated_arg_names
        self.configs = configs
        self.heuristic_type = heuristic_type
        self.custom_kernel = custom_kernel
        self.cuda_kernel_saved = False
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "CachingAutotuner gets %d configs for %s",
                len(self.configs),
                self.fn.__name__,
            )
            for c in self.configs:
                log.debug(c)

        self.launchers = []  # type: ignore[var-annotated]
        self.lock = threading.Lock()
        if os.getenv("TRITON_CACHE_DIR") is None:
            os.environ["TRITON_CACHE_DIR"] = os.path.join(
                cache_dir(),
                "triton",
                str(self.triton_meta.get("device", 0)),
            )
        log.debug("Triton cache dir: %s", os.environ["TRITON_CACHE_DIR"])

        self.size_hints = size_hints
        self.coordesc_tuner = CoordescTuner(
            is_mm=False,
            name=self.fn.__name__,
            size_hints=size_hints,
            inductor_meta=self.inductor_meta,
        )
        self.filename = filename

    def precompile(self, warm_cache_only=False):
        with self.lock:
            if self.launchers:
                return
            self.launchers = []
            compiled_binaries = []
            if not self.configs:
                raise RuntimeError("No triton configs are available")
            for c in self.configs:
                try:
                    compiled_binary, launcher = self._precompile_config(
                        c, warm_cache_only
                    )
                except OutOfResources as e:
                    if len(self.configs) == 1:
                        # There are no valid Triton configs
                        raise e
                    # Skip the config if we run out of resource
                    continue
                self.launchers.append(launcher)
                compiled_binaries.append(compiled_binary)

            if len(self.launchers) == 0:
                raise RuntimeError(
                    "No valid triton configs. Report a fatal compilation error"
                )

            seen_configs = set(self.configs)

            device_prop = self.device_props
            if (
                self.inductor_meta.get("dynamic_scale_rblock", True)
                and self.heuristic_type == HeuristicType.REDUCTION
                and self.size_hints is not None
                # Disable for AMDGPU/Intel as Triton is not ready to return n_regs for a compiled_binary.
                and device_prop.type == "cuda"
                and device_prop.major
                and device_prop.major >= 8
            ):
                assert device_prop.regs_per_multiprocessor
                assert device_prop.max_threads_per_multi_processor
                assert device_prop.multi_processor_count
                for triton_config, compiled_binary in zip(
                    self.configs, compiled_binaries
                ):
                    assert len(self.size_hints) == 2
                    xblock = triton_config.kwargs.get("XBLOCK", 1)
                    rblock = triton_config.kwargs["RBLOCK"]
                    total_block = (self.size_hints[0] + xblock - 1) // xblock
                    nreg = getattr(compiled_binary, "n_regs", None)
                    if nreg is None:
                        continue

                    # make sure rblock is not too small
                    if rblock <= 64:
                        continue

                    # each SM of A100 has 65536 32-bit registers. To maximize
                    # the theoretical occupancy, we need run 2048 threads on each
                    # SM. So each thread should use no more than 65536 / 2048
                    # = 32 registers. In cases where occupancy matters, and each
                    # thread uses too many registers, reduce RBLOCK to reduce
                    # the register usage.
                    # For kernel https://gist.github.com/shunting314/e4cccc031fe30d378b9b23c08c238cbd
                    # from PLBartForCausalLM, latency improve from
                    # 7.795ms to 4.883ms.
                    #
                    if (
                        nreg
                        <= device_prop.regs_per_multiprocessor
                        // device_prop.max_threads_per_multi_processor
                    ):
                        continue

                    nreg_per_warp = nreg * 32
                    nreg_per_block = nreg_per_warp * triton_config.num_warps

                    # Previously we set max_blocks_per_sm to 'max_threads_per_multi_processo / (32 * num_warps)'
                    # The formula below is a tighter upper bound since we have the assumption that
                    #   nreg > device_prop.regs_per_multiprocessor // device_prop.max_threads_per_multi_processor
                    # due to the if condition above and:
                    #   regs_per_multiprocessor / nreg_per_block
                    #   = regs_per_multiprocessor / (nreg * 32 * num_warps)
                    #   < regs_per_multiprocessor / ((regs_per_multiprocessor / max_threads_per_multi_processor) * 32 * num_warps)
                    #   = max_threads_per_multi_processor / (32 * num_warps)
                    # Using a tigher upper bound can reveal more optimization opportunities.
                    max_blocks_per_sm = max(
                        device_prop.regs_per_multiprocessor // nreg_per_block, 1
                    )

                    if (
                        total_block
                        <= max_blocks_per_sm * device_prop.multi_processor_count
                    ):
                        # no need to improve occupancy
                        continue
                    new_config = copy.deepcopy(triton_config)
                    new_config.kwargs["RBLOCK"] = rblock // 2
                    if new_config in seen_configs:
                        continue
                    seen_configs.add(new_config)
                    log.debug(
                        "Dynamically scale down RBLOCK from TritonConfig(%s) and get a new TritonConfig(%s)",
                        triton_config,
                        new_config,
                    )
                    self.launchers.append(
                        self._precompile_config(new_config, warm_cache_only)[1]
                    )
            self.configs = None

    def get_device_interface(self):
        # this code cannot run in compile workers, because it imports from torch
        from torch._dynamo.device_interface import get_interface_for_device

        return get_interface_for_device(self.device_props.type.replace("hip", "cuda"))

    def _precompile_config(self, cfg: Config, warm_cache_only: bool):
        """Ahead of time compile a given autotuner config."""
        compile_meta = copy.deepcopy(self.triton_meta)
        for k, v in cfg.kwargs.items():
            if self.device_props.type != "hip":
                if k == "matrix_instr_nonkdim":
                    compile_meta["matrix_instr_nonkdim"] = v
                    continue
                if k == "waves_per_eu":
                    compile_meta["waves_per_eu"] = v
                    continue
            compile_meta["constants"][self.fn.arg_names.index(k)] = v
        compile_meta["num_warps"] = cfg.num_warps
        compile_meta["num_stages"] = cfg.num_stages
        compile_meta["debug"] = self.inductor_meta.get(
            "assert_indirect_indexing", True
        ) and not self.inductor_meta.get("is_hip", False)

        # device type will be "hip" rather than "cuda" here
        compile_meta["device_type"] = self.device_props.type
        compile_meta["cc"] = self.device_props.cc

        if ASTSource:
            compile_args = (
                ASTSource(
                    self.fn,
                    compile_meta["signature"],
                    compile_meta["constants"],
                    compile_meta["configs"][0],
                ),
            )

            cc_str = str(compile_meta["cc"])
            if "gfx10" in cc_str or "gfx11" in cc_str:
                rocm_warp_size = 32
            else:
                rocm_warp_size = 64

            if GPUTarget:
                target = GPUTarget(
                    compile_meta["device_type"],
                    compile_meta["cc"],
                    rocm_warp_size if torch.version.hip else 32,
                )
            else:
                target = (
                    (compile_meta["device_type"], compile_meta["cc"])
                    if not torch.version.hip
                    else [
                        compile_meta["device_type"],
                        compile_meta["cc"],
                        rocm_warp_size,
                    ]
                )

            options = {
                "num_warps": compile_meta["num_warps"],
                "num_stages": compile_meta["num_stages"],
                "debug": compile_meta["debug"],
            }
            if self.device_props.type != "hip":
                if "waves_per_eu" in compile_meta:
                    options["waves_per_eu"] = compile_meta["waves_per_eu"]
                if "matrix_instr_nonkdim" in compile_meta:
                    options["matrix_instr_nonkdim"] = compile_meta[
                        "matrix_instr_nonkdim"
                    ]
            compile_kwargs = {
                "target": target,
                "options": options,
            }
        else:
            compile_args = (self.fn,)
            compile_kwargs = compile_meta

        if warm_cache_only:
            return (
                triton.compile(*compile_args, **compile_kwargs),
                None,
            )

        # importing from torch is safe now that precompile has returned
        from torch._dynamo.device_interface import DeviceGuard

        device_interface = self.get_device_interface()

        # load binary to the correct device
        with DeviceGuard(device_interface, compile_meta["device"]):  # type: ignore[attr-defined]
            # need to initialize context
            device_interface.synchronize(device_interface.current_device())

            try:
                binary = triton.compile(*compile_args, **compile_kwargs)
            except Exception:
                log.exception(
                    "Triton compilation failed: %s\n%s\nmetadata: %s",
                    self.inductor_meta.get("kernel_name", "triton_"),
                    self.fn.src,
                    compile_meta,
                )
                raise
            binary._init_handles()

        call_args = [
            arg
            for i, arg in enumerate(self.fn.arg_names)
            if i not in self.fn.constexprs
        ]
        def_args = [name for name in self.fn.arg_names if name not in cfg.kwargs]

        binary_shared = (
            binary.shared if hasattr(binary, "shared") else binary.metadata.shared
        )

        scope = {
            "grid_meta": cfg.kwargs,
            "bin": binary,
            "launch_enter_hook": CompiledKernel.launch_enter_hook,
            "launch_exit_hook": CompiledKernel.launch_exit_hook,
            "metadata": binary.packed_metadata
            if hasattr(binary, "packed_metadata")
            else binary.metadata,
            "shared": binary_shared,
        }

        scope["num_warps"] = (
            binary.num_warps
            if hasattr(binary, "num_warps")
            else binary.metadata.num_warps
        )

        scope["cta_args"] = (
            (binary.num_ctas, *get_first_attr(binary, "cluster_dims", "clusterDims"))
            if hasattr(binary, "num_ctas")
            else (
                (binary.metadata.num_ctas, *binary.metadata.cluster_dims)
                if hasattr(binary, "metadata")
                else ()
            )
        )

        scope["function"] = get_first_attr(binary, "function", "cu_function")

        def get_launch_args_without_kernel_launch_metadata(
            grid,
            grid_0,
            grid_1,
            grid_2,
            stream,
            function,
            metadata,
            bin,
            launch_enter_hook,
            launch_exit_hook,
            num_warps,
            shared,
            cta_args,
            args,
        ):
            """
            Construct launch args before CompiledKernel.launch_metadata is added.
            """
            return (
                grid_0,
                grid_1,
                grid_2,
                num_warps,
                *cta_args,
                shared,
                stream,
                function,
                launch_enter_hook,
                launch_exit_hook,
                metadata,
            )

        # Getting the kernel launch args is extremely perf-sensitive.  Evaluating
        # `bin.launch_metadata` is relatively expensive, and returns None unless a
        # `launch_enter_hook` is installed.  So if we don't have that hook installed,
        # we want to burn None in to the launch args with zero overhead.
        # See https://github.com/pytorch/pytorch/issues/123597
        if binary.launch_enter_hook:

            def get_launch_args_with_kernel_launch_metadata(
                grid,
                grid_0,
                grid_1,
                grid_2,
                stream,
                function,
                metadata,
                bin,
                launch_enter_hook,
                launch_exit_hook,
                num_warps,
                shared,
                cta_args,
                args,
            ):
                """
                Construct launch args after CompiledKernel.launch_metadata is added
                by https://github.com/openai/triton/pull/3492 .
                """
                return (
                    grid_0,
                    grid_1,
                    grid_2,
                    stream,
                    function,
                    metadata,
                    bin.launch_metadata(grid, stream, *args),
                    launch_enter_hook,
                    launch_exit_hook,
                )

        else:

            def get_launch_args_with_kernel_launch_metadata(
                grid,
                grid_0,
                grid_1,
                grid_2,
                stream,
                function,
                metadata,
                bin,
                launch_enter_hook,
                launch_exit_hook,
                num_warps,
                shared,
                cta_args,
                args,
            ):
                """
                Construct launch args after CompiledKernel.launch_metadata is added
                by https://github.com/openai/triton/pull/3492 .
                """
                return (
                    grid_0,
                    grid_1,
                    grid_2,
                    stream,
                    function,
                    metadata,
                    None,
                    launch_enter_hook,
                    launch_exit_hook,
                )

        scope["get_launch_args"] = (
            get_launch_args_with_kernel_launch_metadata
            if hasattr(binary, "launch_metadata")
            else get_launch_args_without_kernel_launch_metadata
        )

        scope["runner"] = get_first_attr(binary, "run", "c_wrapper")

        exec(
            f"""
            def launcher({', '.join(def_args)}, grid, stream):
                if callable(grid):
                    grid_0, grid_1, grid_2 = grid(grid_meta)
                else:
                    grid_0, grid_1, grid_2 = grid

                args = {', '.join(call_args)},
                launch_args = get_launch_args(
                    grid, grid_0, grid_1, grid_2, stream, function,
                    metadata, bin, launch_enter_hook, launch_exit_hook,
                    num_warps, shared, cta_args, args
                )
                runner(*launch_args, *args)
                return bin
            """.lstrip(),
            scope,
        )

        launcher = scope["launcher"]
        launcher.config = cfg
        launcher.n_regs = getattr(binary, "n_regs", None)
        launcher.n_spills = getattr(binary, "n_spills", None)
        launcher.shared = binary_shared
        launcher.store_cubin = self.inductor_meta.get("store_cubin", False)
        # store this global variable to avoid the high overhead of reading it when calling run
        if launcher.store_cubin:
            launcher.fn = self.fn
            launcher.bin = binary

        return binary, launcher

    def bench(self, launcher, *args, grid, **kwargs):
        """Measure the performance of a given launcher"""
        # we don't skip configs wiht spilled registers when auto-tuning custom
        # (user-written) Triton kernels, as (i) we don't have any knowledge or
        # control over the kernel code; (ii) there is empirical evidence that
        # for some (complicated) custom Triton kernels, a register-spilling
        # config may yield the best latency.
        if not self.custom_kernel and launcher.n_spills > self.inductor_meta.get(
            "spill_threshold", 16
        ):
            log.debug(
                "Skip config %s because of register spilling: %d",
                launcher.config,
                launcher.n_spills,
            )
            return float("inf")

        device_interface = self.get_device_interface()
        stream = device_interface.get_raw_stream(  # type: ignore[call-arg]
            device_interface.current_device()
        )

        def kernel_call():
            if launcher.config.pre_hook is not None:
                launcher.config.pre_hook(
                    {**dict(zip(self.arg_names, args)), **launcher.config.kwargs}
                )

            cloned_args, cloned_kwargs = self.clone_args(*args, **kwargs)
            launcher(
                *cloned_args,
                **cloned_kwargs,
                grid=grid,
                stream=stream,
            )

        return do_bench_gpu(kernel_call, rep=40, fast_flush=True)

    def clone_args(self, *args, **kwargs) -> Tuple[List[Any], Dict[str, Any]]:
        from ..compile_fx import clone_preserve_strides

        # clone inplace buffers to avoid autotune contaminating them if
        # the kernel does in-place stores. avoid cloning other buffers because
        # it leads to increase memory use
        cloned_args = []
        for i, arg in enumerate(args):
            if self.fn.arg_names[i] in self.mutated_arg_names:
                assert isinstance(arg, torch.Tensor)
                cloned_args.append(clone_preserve_strides(arg))
            else:
                cloned_args.append(arg)

        cloned_kwargs: Dict[str, Any] = {}
        for name, arg in kwargs.items():
            if name in self.mutated_arg_names:
                assert isinstance(arg, torch.Tensor)
                cloned_kwargs[name] = clone_preserve_strides(arg)
            else:
                cloned_kwargs[name] = arg

        return cloned_args, cloned_kwargs

    @dynamo_timed
    def benchmark_all_configs(self, *args, **kwargs):
        timings = {
            launcher: self.bench(launcher, *args, **kwargs)
            for launcher in self.launchers
        }

        for k, v in timings.items():
            self.coordesc_tuner.cache_benchmark_result(k.config, v)

        if log.isEnabledFor(logging.DEBUG):
            log.debug("Benchmark all input configs for %s, get:", self.fn.__name__)
            for k, v in timings.items():
                log.debug(
                    "%s: %f, nreg %d, nspill %d, #shared-mem %s",
                    k.config,
                    v,
                    k.n_regs,
                    k.n_spills,
                    k.shared,
                )

        return timings

    def autotune_to_one_config(self, *args, **kwargs):
        """Do the actual autotuning"""
        start_time = time.time_ns()
        timings = self.benchmark_all_configs(*args, **kwargs)
        time_taken_ns = time.time_ns() - start_time
        self.launchers = [builtins.min(timings, key=timings.get)]
        if self.save_cache_hook:
            self.save_cache_hook(self.launchers[0].config, time_taken_ns)

    def save_cuda_kernel(self, grid, stream, launcher):
        if callable(grid):
            grid_x, grid_y, grid_z = grid(launcher.config.kwargs)
        else:
            grid_x, grid_y, grid_z = grid

        key = self.inductor_meta.get("kernel_name", None)  # unique kernel name
        assert key is not None, "kernel_name can not be None"
        params = {
            "mangled_name": launcher.bin.metadata.name
            if hasattr(launcher.bin.metadata, "name")
            else launcher.bin.metadata["name"],
            "grid_x": grid_x,
            "grid_y": grid_y,
            "grid_z": grid_z,
            "x_block": launcher.config.kwargs.get("XBLOCK", 1),
            "y_block": launcher.config.kwargs.get("YBLOCK", None),
            "z_block": launcher.config.kwargs.get("ZBLOCK", None),
            "num_warps": launcher.bin.num_warps
            if hasattr(launcher.bin, "num_warps")
            else launcher.bin.metadata.num_warps,
            "shared_mem": launcher.bin.shared
            if hasattr(launcher.bin, "shared")
            else launcher.bin.metadata.shared,
            "stream": stream,
            # User defined triton kernels will have arbitrary kwarg names
            "meta": launcher.config.kwargs,
        }
        from torch._inductor.codecache import CudaKernelParamCache

        binary = (
            launcher.bin.asm["cubin"]
            if self.device_props.type != "hip"
            else launcher.bin.asm["hsaco"]
        )
        CudaKernelParamCache.set(key, params, binary)

        self.cuda_kernel_saved = True

    def coordinate_descent_tuning(self, launcher, *args, **kwargs):
        """
        Coordinate descent tuning can be run with or without max-autotune.

        The only difference between these two is the starting config for coordinate_descent tuning.
        E.g., assuming regular autotune only get one config C1; while max-autotune get 4 configs C1, C2, C3, C4
        and max-autotune figure out C3 is the best.

        Then if coordinate desecnt tuning is run with max-autotune disabled, it will start from C1;
        while if coordinate descent tuning is run with max-autotune enabled, it will start from C3.
        """
        if (
            self.heuristic_type == HeuristicType.TEMPLATE
            or self.heuristic_type == HeuristicType.USER_AUTOTUNE
        ):
            # skip triton template
            return launcher

        config2launcher = {launcher.config: launcher}

        def benchmark_one_config(config):
            with self.lock:
                _, launcher = self._precompile_config(config, False)
            config2launcher[config] = launcher

            out = self.bench(launcher, *args, **kwargs)
            log.debug(
                "COORDESC: %s: %f, nreg %d, nspill %d, #shared-mem %d",
                launcher.config,
                out,
                launcher.n_regs,
                launcher.n_spills,
                launcher.shared,
            )
            return out

        assert not (
            self.heuristic_type == HeuristicType.PERSISTENT_REDUCTION
            and "RBLOCK" in launcher.config.kwargs
        ), "Coordinate descent tuner relies on the assumption that persistent reduction's triton config does not have RBLOCK"
        start_time = time.time_ns()
        best_config = self.coordesc_tuner.autotune(
            benchmark_one_config, launcher.config, None
        )
        time_taken_ns = time.time_ns() - start_time
        best_config.found_by_coordesc = True

        if self.save_cache_hook:
            self.save_cache_hook(best_config, time_taken_ns, found_by_coordesc=True)
        return config2launcher.get(best_config)

    def run(self, *args, grid, stream, **kwargs):
        if len(self.launchers) != 1:
            if len(self.launchers) == 0:
                self.precompile()
            if len(self.launchers) > 1:
                self.autotune_to_one_config(*args, grid=grid, **kwargs)

        if not getattr(
            self.launchers[0].config, "found_by_coordesc", False
        ) and self.inductor_meta.get("coordinate_descent_tuning", False):
            self.launchers = [
                self.coordinate_descent_tuning(
                    self.launchers[0], *args, grid=grid, **kwargs
                )
            ]

        (launcher,) = self.launchers
        if launcher.store_cubin:
            self.save_cuda_kernel(grid, stream, launcher)

        if launcher.config.pre_hook is not None:
            launcher.config.pre_hook(
                {**dict(zip(self.arg_names, args)), **launcher.config.kwargs, **kwargs}
            )

        if os.environ.get("TORCHINDUCTOR_DUMP_LAUNCH_PARAMS", 0) == "1":
            _dump_launch_params(args, kwargs, launcher, self.fn.__name__)

        # it is faster than entering and exiting a context manager, even if the context
        # manager is a nullcontext.
        if autograd_profiler._is_profiler_enabled:
            # grid can be a tuple of ints or a string.
            if isinstance(grid, tuple):
                grid_info = str(grid)
            else:
                grid_info = getattr(grid, "grid_fn_str", "")
            with torch._C._profiler._RecordFunctionFast(
                self.inductor_meta.get("kernel_name", "triton kernel"),
                args,
                {
                    "kernel_file": "" if self.filename is None else self.filename,
                    "kernel_backend": "triton",
                    "grid": grid_info,
                    "stream": stream,
                },
            ):
                return launcher(
                    *args,
                    **kwargs,
                    grid=grid,
                    stream=stream,
                )
        else:
            return launcher(
                *args,
                **kwargs,
                grid=grid,
                stream=stream,
            )

