class FlopCounterMode(TorchDispatchMode):
    """
    ``FlopCounterMode`` is a context manager that counts the number of flops within its context.

    It does this using a ``TorchDispatchMode``.

    It also supports hierarchical output by passing a module (or list of
    modules) to FlopCounterMode on construction. If you do not need hierarchical
    output, you do not need to use it with a module.

    Example usage

    .. code-block:: python

        mod = ...
        with FlopCounterMode(mod) as flop_counter:
            mod.sum().backward()

    """

    def __init__(
            self,
            mods: Optional[Union[torch.nn.Module, List[torch.nn.Module]]] = None,
            depth: int = 2,
            display: bool = True,
            custom_mapping: Optional[Dict[Any, Any]] = None):
        self.flop_counts: Dict[str, Dict[Any, int]] = defaultdict(lambda: defaultdict(int))
        self.depth = depth
        self.display = display
        if custom_mapping is None:
            custom_mapping = {}
        if mods is not None:
            warnings.warn("mods argument is not needed anymore, you can stop passing it", stacklevel=2)
        self.flop_registry = {
            **flop_registry,
            **{k: v if getattr(v, "_get_raw", False) else shape_wrapper(v) for k, v in custom_mapping.items()}
        }
        self.mod_tracker = ModuleTracker()

    def get_total_flops(self) -> int:
        return sum(self.flop_counts['Global'].values())

    def get_flop_counts(self) -> Dict[str, Dict[Any, int]]:
        """Return the flop counts as a dictionary of dictionaries.

        The outer
        dictionary is keyed by module name, and the inner dictionary is keyed by
        operation name.

        Returns:
            Dict[str, Dict[Any, int]]: The flop counts as a dictionary.
        """
        return {k: dict(v) for k, v in self.flop_counts.items()}

    def get_table(self, depth=None):
        if depth is None:
            depth = self.depth
        if depth is None:
            depth = 999999

        import tabulate
        tabulate.PRESERVE_WHITESPACE = True
        header = ["Module", "FLOP", "% Total"]
        values = []
        global_flops = self.get_total_flops()
        global_suffix = get_suffix_str(global_flops)
        is_global_subsumed = False

        def process_mod(mod_name, depth):
            nonlocal is_global_subsumed

            total_flops = sum(self.flop_counts[mod_name].values())

            is_global_subsumed |= total_flops >= global_flops

            padding = " " * depth
            values = []
            values.append([
                padding + mod_name,
                convert_num_with_suffix(total_flops, global_suffix),
                convert_to_percent_str(total_flops, global_flops)
            ])
            for k, v in self.flop_counts[mod_name].items():
                values.append([
                    padding + " - " + str(k),
                    convert_num_with_suffix(v, global_suffix),
                    convert_to_percent_str(v, global_flops)
                ])
            return values

        for mod in sorted(self.flop_counts.keys()):
            if mod == 'Global':
                continue
            mod_depth = mod.count(".") + 1
            if mod_depth > depth:
                continue

            cur_values = process_mod(mod, mod_depth - 1)
            values.extend(cur_values)

        # We do a bit of messing around here to only output the "Global" value
        # if there are any FLOPs in there that aren't already fully contained by
        # a module.
        if 'Global' in self.flop_counts and not is_global_subsumed:
            for idx, value in enumerate(values):
                values[idx][0] = " " + values[idx][0]

            values = process_mod('Global', 0) + values

        if len(values) == 0:
            values = [["Global", "0", "0%"]]

        return tabulate.tabulate(values, headers=header, colalign=("left", "right", "right"))

    def __enter__(self):
        self.flop_counts.clear()
        self.mod_tracker.__enter__()
        super().__enter__()
        return self

    def __exit__(self, *args):
        super().__exit__(*args)
        self.mod_tracker.__exit__()
        if self.display:
            print(self.get_table(self.depth))

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs else {}
        out = func(*args, **kwargs)
        return self._count_flops(func._overloadpacket, out, args, kwargs)

    def _count_flops(self, func_packet, out, args, kwargs):
        if func_packet in self.flop_registry:
            flop_count_func = self.flop_registry[func_packet]
            flop_count = flop_count_func(*args, **kwargs, out_val=out)  # type: ignore[operator]
            for par in set(self.mod_tracker.parents):
                self.flop_counts[par][func_packet] += flop_count

        return out

