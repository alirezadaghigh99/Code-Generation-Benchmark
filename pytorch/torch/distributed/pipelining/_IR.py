class Pipe(torch.nn.Module):
    def __init__(
        self,
        split_gm: fx.GraphModule,
        num_stages: int,
        has_loss_and_backward: bool,
        loss_spec,
    ):
        # TODO: is there a way not to hard wire init?
        torch.nn.Module.__init__(self)
        self.split_gm: fx.GraphModule = split_gm
        self.executor: DetachExecutor = DetachExecutor(self.split_gm)
        self.num_stages: int = num_stages
        self.has_loss_and_backward = has_loss_and_backward
        self.loss_spec = loss_spec

        for node in split_gm.graph.nodes:
            assert (
                node.op in {"call_module", "placeholder", "output"}
                or (node.op, node.target) == ("call_function", operator.getitem)
                or (node.op, node.target) == ("call_method", "backward")
                or (node.op, node.target) == ("call_function", stage_backward)
                or (node.op, node.target)
                == ("call_function", _null_coalesce_accumulate)
            ), node

        # Detect replicated parameters so we know that we have to do an additional allreduce
        # before applying the optimizer
        #
        # Note that this also handles the case where there were multiple calls to a single
        # module from different stages, regardless of whether that module invocation
        # was handled by the logic above.

        # Map parameter value to a dictionary that maps the user pipeline module
        # to the local qualname within that module
        params_to_users: Dict[torch.nn.Parameter, Dict[str, str]] = {}

        for m_qualname, mod in self.split_gm.named_children():
            for p_qualname, param in mod.named_parameters():
                params_to_users.setdefault(param, {})
                params_to_users[param][m_qualname] = p_qualname

        self.replicated_params: List[Dict[str, str]] = [
            use_mapping
            for _, use_mapping in params_to_users.items()
            if len(use_mapping) > 1
        ]

        # We must break the aliasing relationship between the replicated parameters for correct
        # numerics in reference runs. If we do not do this, the autograd tape in separate stages
        # will have a reference to the same tensor value and will erroneously apply gradient
        # updates multiple times. Therefore, for each replicated parameter set, we deepcopy the
        # values so that we have separate instances.
        for param_mapping in self.replicated_params:
            for submod_name, param_qualname in param_mapping.items():
                submod = getattr(self.split_gm, submod_name)
                atoms = param_qualname.split(".")
                for atom in atoms[:-1]:
                    submod = getattr(submod, atom)
                setattr(submod, atoms[-1], copy.deepcopy(getattr(submod, atoms[-1])))

        def throw(self, *args, **kwargs):
            raise RuntimeError(
                "To run pipeline locally, invoke the Pipe object directly, not `split_gm`"
            )

        self.split_gm.forward = throw

        # Make submodules use custom direct-serialized GraphModule
        i = 0
        while True:
            try:
                name = f"submod_{i}"
                submod = getattr(self.split_gm, name)
                submod.__class__.__reduce__ = _direct_serialization_reduce
                i += 1
            except AttributeError:
                break

    def forward(self, *args, **kwargs):
        executor_args = args
        if len(kwargs) > 0:
            parameters = []
            for node in self.split_gm.graph.nodes:
                if node.op == "placeholder":
                    if node.args and len(node.args) > 0:
                        parameters.append(
                            Parameter(
                                node.target,
                                Parameter.POSITIONAL_OR_KEYWORD,
                                default=node.args[0],
                            )
                        )
                    else:
                        parameter_kind = Parameter.POSITIONAL_OR_KEYWORD
                        param_name = node.target
                        if node.target.startswith("**"):
                            parameter_kind = Parameter.VAR_KEYWORD  # type: ignore[assignment]
                            param_name = param_name[2:]
                        elif node.target.startswith("*"):
                            parameter_kind = Parameter.VAR_POSITIONAL  # type: ignore[assignment]
                            param_name = param_name[1:]
                        parameters.append(Parameter(param_name, parameter_kind))
            signature = Signature(parameters)
            ba = signature.bind(*args, **kwargs)
            ba.apply_defaults()
            executor_args = ba.arguments.values()  # type: ignore[assignment]

        res = self.executor.run(*executor_args)

        return res

    def get_stage_module(self, stage_idx: int) -> torch.nn.Module:
        """
        Return a stage module corresponding to `stage_idx` of the `pipe`.
        """
        if stage_idx < 0 or stage_idx >= self.num_stages:
            raise ValueError(f"Invalid stage index {stage_idx}!")
        return getattr(self.split_gm, f"submod_{stage_idx}")

    @staticmethod
    def _number_and_count_forward_stages(gm: fx.GraphModule):
        num_stages = 0
        found_idxs: Dict[int, None] = {}
        for node in gm.graph.nodes:
            if node.op == "call_module" and node.target.startswith("submod_"):
                node.meta["stage_idx"] = int(node.target[len("submod_") :])
                found_idxs.setdefault(node.meta["stage_idx"])
                num_stages += 1

        # this assert will fail if a split point is inserted before the first layer, which creates empty first submodule
        # Update: the following assert may fail against some torch versions >=
        # 2.2.0, as:
        # submod_0, submod_1, submod_2, ...
        # may be named as
        # submod_0, submod_2, submod_4, ...
        # TODO: investigate
        # assert all(i in found_idxs for i in range(num_stages))

        return num_stages

    @staticmethod
    def _from_traced(
        mod: torch.nn.Module,
        exported_program: ExportedProgram,
        multi_use_param_spec: Optional[MultiUseParamSpec] = None,
        output_loss_value_spec=None,
        split_policy: Optional[
            Callable[[torch.fx.GraphModule], torch.fx.GraphModule]
        ] = None,
    ):
        """
        Additionally, the ``output_loss_value_spec`` value can be specified to disambiguate
        which value in the output of `forward` is the loss value on which PiPPy should apply
        backpropagation. For example, if your ``forward`` returns a tuple ``(loss, model_out)``,
        you can specify ``output_loss_value_spec=(True, False)``. Or, if your ``forward`` returns
        a dict ``{'loss': loss_value, 'model_out': model_out}``, you can specify
        ``output_loss_value_spec={'loss': True, 'model_out': False}``
        """

        traced = exported_program.module()

        if split_policy is not None:
            logger.info("Auto-splitting model")
            traced = split_policy(traced)  # type: ignore[arg-type]

        logger.debug(traced.print_readable(print_output=False))

        # Deduplicate `get_attr` nodes that refer to the same parameter . Downstream code for moving
        # parameters relies on the invariant that parameter accesses happen once. This is not necessarily
        # the case (especially with custom tracers), so fix that up here.
        get_attr_nodes: Dict[str, fx.Node] = {}
        for node in traced.graph.nodes:
            if node.op == "get_attr":
                get_attr_nodes.setdefault(node.target, node)

                if get_attr_nodes[node.target] != node:
                    node.replace_all_uses_with(get_attr_nodes[node.target])
                    traced.graph.erase_node(node)

        # avoid looking at next node by keeping track of previous pipe_split
        prev_pipe_split_idx = -1
        pipe_split_nodes_to_erase = set()
        for i, node in enumerate(traced.graph.nodes):
            if (node.op, node.target) == ("call_function", pipe_split):
                if prev_pipe_split_idx == i - 1:
                    pipe_split_nodes_to_erase.add(node)
                prev_pipe_split_idx = i

        for node in pipe_split_nodes_to_erase:
            traced.graph.erase_node(node)

        traced.recompile()

        part_idx = 0

        def split_callback(n: fx.Node):
            nonlocal part_idx
            if (n.op, n.target) == (
                "call_function",
                aten_pipe_split_alias,
            ):
                logger.debug(f"Found pipe_split {part_idx}")  # noqa: G004
                part_idx += 1
            return part_idx

        # TODO: what does split do with module invocations? does it move the modules
        # into the submodules?
        split = split_module(traced, mod, split_callback)
        # a (custom) tracer can produce dead code like orphan get_attr nodes
        split.graph.eliminate_dead_code()

        # peephole to remove pipe_split
        for submodule in split.modules():
            if isinstance(submodule, fx.GraphModule):
                for node in submodule.graph.nodes:
                    if (node.op, node.target) == (
                        "call_function",
                        aten_pipe_split_alias,
                    ):
                        submodule.graph.erase_node(node)
                submodule.recompile()

        for name, submodule in split.named_children():
            if isinstance(submodule, fx.GraphModule):
                new_submod = _outline_submodules(submodule.graph)
                # Replace old submod
                split.register_module(name, new_submod)

        # TODO: backport this into split_module
        def delete_user_reference(node, user):
            """
            Delete reference of `node` from `user`'s arg list.
            Args:
                - node: a `get_attr` node at root.
                - user: a submodule node that uses `node`.
            """
            assert len(user.kwargs) == 0
            use_idxs = [i for i, arg in enumerate(user.args) if arg == node]
            assert len(use_idxs) == 1
            args_copy = list(user.args)
            args_copy.pop(use_idxs[0])
            user.args = tuple(args_copy)
            logger.debug(
                f"Deleted {node} from user {user}, arg index = {use_idxs[0]}"  # noqa: G004
            )

        # A list of param referrals for deferred deletion.
        # To be accumulated in `move_param_to_callee`.
        to_delete = list()

        def _recursive_getattr_with_parent(mod, fqn):
            # Returns getattr call given a nested FQN, and the last parent
            atoms = fqn.split(".")
            for atom in atoms[:-1]:
                if not hasattr(mod, atom):
                    return None, None
                mod = getattr(mod, atom)
            if not hasattr(mod, atoms[-1]):
                return mod, None
            attr = getattr(mod, atoms[-1])
            return mod, attr

        def move_param_to_callee(
            root,
            callee_name,
            param_fqn,
        ):
            """
            Move a parameter from the root module to a submodule.
            Args:
                root: The root module.
                callee_name: The name of the submodule to move the parameter to.
                param_fqn: The fully qualified name of the parameter to move.
            """
            # `atoms` is a list of strings representing the path to the
            # parameter in the original model
            atoms = param_fqn.split(".")
            mod_itr, param_val = _recursive_getattr_with_parent(split, param_fqn)
            # Check whether the parameter is a buffer or a parameter
            is_buffer = atoms[-1] in mod_itr._buffers

            # Check whether the parameter is a tensor
            assert isinstance(param_val, torch.Tensor), (
                f"Expected '{param_fqn}' to be {torch.Tensor} but got {type(param_val)}."
                + (
                    f" It might happen if module '{param_fqn}' was passed to some 'leaf function'"
                    f"(see https://pytorch.org/docs/stable/fx.html#fx.wrap). Please inspect "
                    f"usages of '{param_fqn}' in the traced graph."
                    if isinstance(param_val, torch.nn.Module)
                    else ""
                )
            )

            # Get submodule
            callee = root.get_submodule(callee_name)
            assert not hasattr(
                callee, param_fqn
            ), f"Module {callee_name} already has a parameter named {param_fqn}"

            # Assign the parameter to the submodule
            if is_buffer:
                _assign_attr(
                    param_val,
                    callee,
                    param_fqn,
                    attr_kind=_AttrKind.BUFFER,
                    persistent=True,  # TODO: handle non-persistent buffer
                )
            else:
                _assign_attr(
                    param_val,
                    callee,
                    param_fqn,
                    attr_kind=_AttrKind.PARAMETER,
                )
            logger.debug(f"Moved parameter {param_fqn} to {callee_name}")  # noqa: G004

            # Next step is to replace placeholder of submodule with a get_attr.
            # Those placeholders are created by `split_module` inside each
            # submodule.
            # Update: this step is now moved to `_sink_params` because
            # `_sink_params` can do it recursively (i.e. for modules inside
            # submodule)

            to_delete.append((mod_itr, atoms[-1]))

        # Get the list of all parameters in the root module
        attr_nodes = list(filter(lambda n: n.op == "get_attr", split.graph.nodes))
        for node in attr_nodes:
            # Check whether the parameter is used in only one submodule
            if len(node.users) > 1:
                logger.info(
                    f"Parameter {node.target} used in multiple stages: {node.users}."  # noqa: G004
                )
            for user in node.users:
                assert user.op == "call_module"
                # Move parameter into submodule
                move_param_to_callee(
                    split,
                    user.target,
                    node.target,
                )

        # [aliasing] store tensor id -> list of FQNs, built from state dict
        # Also assign non-persistent buffers
        id_to_fqns: Dict[int, Set[str]] = defaultdict(set)
        for fqn, tensor in mod.state_dict(keep_vars=True).items():
            id_to_fqns[id(tensor)].add(fqn)
        for fqn, tensor in mod.named_buffers():
            id_to_fqns[id(tensor)].add(fqn)

        # After moving the params to their corresponding hierarchies, we also
        # need to move the `get_attr` nodes from the root of the graph to those
        # hierarchies.
        # [aliasing] use id -> fqn mapping to list out all valid FQNs
        inputs_to_state: Dict[str, List[str]] = {}
        for attr in attr_nodes:
            _, tensor = _recursive_getattr_with_parent(mod, attr.target)
            fqns = list(id_to_fqns[id(tensor)])
            if fqns:
                inputs_to_state[attr.name] = fqns
            elif attr.target in exported_program.constants:  # lifted constants
                inputs_to_state[attr.name] = [attr.target]

        # [aliasing] for each submodule split, assign attributes on FQNs that may be used.
        # We determine this based on whether or not the FQN attribute parent exists.
        # i.e. if the last submodule exists, assign the attribute.
        added_attributes: Dict[str, List[str]] = defaultdict(list)
        for fqn, tensor in mod.state_dict(keep_vars=True).items():
            for name, submod in split.named_children():
                if isinstance(submod, fx.GraphModule):
                    parent, child = _recursive_getattr_with_parent(submod, fqn)
                    if (
                        parent and child is None
                    ):  # parent exists, attribute doesn't -> assign
                        added_attributes[name].append(fqn)
                        setattr(parent, fqn.split(".")[-1], tensor)

        # Deferral deletion: Remove the original attributes (to params) from the
        # root GraphModule
        for mod_itr, last_atom in to_delete:
            try:
                delattr(mod_itr, last_atom)
            except AttributeError:
                # This is expected if the parameter is used in multiple stages
                pass

        # This is done by (1) `_sink_params` at each submodule;
        for name, submod in split.named_children():
            if isinstance(submod, fx.GraphModule):
                _sink_params(submod, inputs_to_state, [])
                submod.graph.lint()
                submod.recompile()

        # [aliasing] This step is not super necessary, but helps reduce parameter usage/memory.
        # After _sink_params() routine has run, clean up unused attributes that we previously added.
        # Determine this based on the get_attr nodes - if not used, remove it.
        for name, attributes in added_attributes.items():
            submod = getattr(split, name)
            unused_attributes = set(attributes)
            # track used attributes in the submodule, running DFS on subgraph hierarchy
            stack = [("", submod)]  # (scope, submodule)
            while stack:
                scope, _mod = stack.pop()
                if isinstance(_mod, (fx.GraphModule, InterpreterModule)):
                    for node in _mod.graph.nodes:
                        if node.op == "get_attr":
                            # get_attr might get access deeper level attribute
                            fqn = scope + "." + node.target if scope else node.target
                            if fqn in unused_attributes:  # used, remove it
                                unused_attributes.remove(fqn)
                for _name, _submod in _mod.named_children():
                    stack.append((scope + "." + _name if scope else _name, _submod))
            # delete unused attributes
            for attr in unused_attributes:
                mod_itr, atoms = submod, attr.split(".")
                for atom in atoms[:-1]:
                    mod_itr = getattr(mod_itr, atom)
                delattr(mod_itr, atoms[-1])

        for node in attr_nodes:
            # And (2): remove `get_attr` node from submod's arg list
            for user in copy.copy(node.users):
                assert user.op == "call_module"
                delete_user_reference(node, user)
            # And (3): remove the `get_attr` node from the root graph.
            split.graph.erase_node(node)

        split.delete_all_unused_submodules()
        split.graph.lint()
        split.recompile()

        num_stages = Pipe._number_and_count_forward_stages(split)

        has_loss_and_backward = False
        generated_loss_spec = output_loss_value_spec

        if output_loss_value_spec is not None:
            loss_node, output_node, generated_loss_spec = _find_loss_output(
                mod, split.graph, output_loss_value_spec
            )
            if loss_node is not None:
                _insert_stage_symbolic_backward(
                    split.graph,
                    loss_node,
                    output_node,
                )
                split.recompile()
                has_loss_and_backward = True
                logger.debug("Pipeline is in training mode, backward pass generated")
            else:
                raise RuntimeError(
                    f"Did not find any loss value according to {output_loss_value_spec=}"
                )
        else:
            logger.debug("Pipeline is in inference mode, backward pass not generated")

        logger.debug("Full pipe model:\n" f"{split}")  # noqa: G004

        return Pipe(
            split,
            num_stages,
            has_loss_and_backward,
            generated_loss_spec,
        )

    def print_readable(self):
        """
        Print the pipe in a human-readable format.
        This will print both the root pipe and each stage module.
        """
        self.split_gm.print_readable()

    @staticmethod
    def _trace_with_export(
        mod: torch.nn.Module,
        example_args: Tuple[Any, ...],
        example_kwargs: Optional[Dict[str, Any]] = None,
    ) -> ExportedProgram:
        logger.info("Tracing model ...")
        try:
            ep = torch.export.export(
                mod,
                example_args,
                example_kwargs,
            )
        except Exception as e:
            raise RuntimeError(
                "It seems that we cannot capture your model as a full graph. "
                "Typical reasons include graph breaks, data/shape-dependent "
                "control flow, or missing meta kernels for custom operators. "
                "You can use our manual pipeline interfaces, or try to fix the "
                "graph breaks, see https://pytorch.org/docs/stable/export.html"
            ) from e

        return ep

    @staticmethod
    def from_tracing(
        mod: torch.nn.Module,
        example_args: Tuple[Any, ...],
        example_kwargs: Optional[Dict[str, Any]] = None,
        split_policy: Optional[Callable[[fx.GraphModule], fx.GraphModule]] = None,
    ):
        # If a param will be used in multiple pipeline stages, we default the strategy to REPLICATE'ing the param across
        # stages instead of TRANSMIT'ting it
        multi_use_param_spec = MultiUseParameterConfig.REPLICATE

        # Figure out which output is loss from output_chunk_spec
        output_loss_value_spec: Any = None
        # Deprecated
        """
        if output_chunk_spec is not None:
            output_loss_value_spec = map_aggregate(
                output_chunk_spec, lambda v: isinstance(v, _LossReducer)
            )
        """

        # Trace with export
        exported_program = Pipe._trace_with_export(
            mod,
            example_args,
            example_kwargs,
        )

        pipe = Pipe._from_traced(
            mod,
            exported_program,
            multi_use_param_spec,
            output_loss_value_spec=output_loss_value_spec,
            split_policy=split_policy,
        )

        # Users want the first pipeline stage to accept kwargs if the original
        # program does. This is controlled by the `_codegen` field of the graph,
        # so we make a copy here. Note: we only want the input spec and not the
        # output spec, because the output spec is for the last stage. Maybe a
        # TODO? Not sure yet.
        split = pipe.split_gm
        traced = exported_program.module()
        submod0 = next(iter(split.children()))
        submod0_sign = signature(submod0.forward)
        model_sign = signature(traced.forward)
        if len(model_sign.parameters) != len(submod0_sign.parameters):
            # We don't change the signature of the first stage if it takes
            # different number of args than original model
            logger.info(
                f"Original model takes {len(model_sign.parameters)} args but the "  # noqa: G004
                f"first pipeline stage takes {len(submod0_sign.parameters)}. "
                "Please provide args to respective pipeline stages."
            )
        else:
            # Support kwargs for the first stage
            submod0.graph._codegen = copy.deepcopy(traced.graph._codegen)
            # `_replace` is actually not "private" or internal. based on this doc:
            # To prevent conflicts with field names, the method and attribute names
            # start with an underscore
            submod0.graph._codegen.pytree_info = (
                submod0.graph._codegen.pytree_info._replace(out_spec=None)
            )
            submod0.recompile()

        return pipe

    def __str__(self):
        return self.split_gm.__str__()

    def __repr__(self):
        return self.split_gm.__repr__()

    def info(self) -> PipeInfo:
        """
        Get information about the pipe.

        Returns
        -------
        PipeInfo
            A dataclass containing information about the pipe.
        """
        return PipeInfo(
            graph=self.split_gm.graph,
            num_stages=self.num_stages,
            has_loss_and_backward=self.has_loss_and_backward,
        )

    def build_stage(
        self,
        stage_index: int,
        device: torch.device,
        group: Optional[ProcessGroup] = None,
    ) -> _PipelineStage:
        """
        Create a `PipelineStage` given a stage index and distributed group.
        The `PipelineStage` can run with `PipelineSchedule`s.
        """
        # Find stage module
        stage_module = self.get_stage_module(stage_index)

        # Move ops argument to device
        # Today PT2 tracer does not treat `x.device` as a symbolic device;
        # instead, the device of tracing time got burned into the generated
        # code.  Here we provide a workaround for users to manually modify the
        # "device" kwarg of operations. Such operation may include:
        # `torch.ones`, `torch.zeros`, `torch.rand`, etc.
        if isinstance(stage_module, torch.fx.GraphModule):
            _modify_graph_op_device(stage_module, device)
        else:
            logger.warning(
                f"Expected a `torch.fx.GraphModule` but got {type(stage_module)}"  # noqa: G004
            )

        # Detach pipe info
        # Note: be careful what's included in `pipe_info`. We don't want to keep
        # a reference to `Pipe` or `Pipe.split_gm` which stops python from
        # recycling them. When python recycles them, other stage modules (which
        # are irrelevant to current rank) can be automatically freed.
        pipe_info = self.info()
        return _PipelineStage(stage_module, stage_index, pipe_info, device, group)

