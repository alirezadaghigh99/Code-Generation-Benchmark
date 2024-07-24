def insert_at_point(
        self,
        point: PTInsertionPoint,
        fn: Callable,
        hooks_group_name: Optional[str] = DEFAULT_HOOKS_GROUP_NAME,
    ) -> List[HookHandle]:
        """
        Inserts given function to the point in the NNCFNetwork, creates hook handle for the inserted function and
        stores created hook handle in a group with the given name. A group name could be used late
        to remove all hooks from the NNCFNetwork which belongs to the group.

        :param point: Target point to insert function.
        :param fn: Function to insert to the NNCFNetwork.
        :param hooks_group_name: Name of hooks group for hook handle associated with the inserted function.
        :return: Hook handle associated with the inserted function.
        """
        handle = None
        if point.insertion_type == PTInsertionType.OPERATOR_PRE_HOOK:
            handle = self._compressed_context.register_pre_hook(fn, point.op_address, point.input_port_id)
        elif point.insertion_type == PTInsertionType.OPERATOR_POST_HOOK:
            handle = self._compressed_context.register_post_hook(fn, point.op_address)
        elif point.insertion_type in [PTInsertionType.NNCF_MODULE_PRE_OP, PTInsertionType.NNCF_MODULE_POST_OP]:
            nncf_module = self.get_module_by_scope(point.module_scope)
            if not isinstance(nncf_module, _NNCFModuleMixin):
                raise nncf.ValidationError(
                    f"Failed to insert pre/post op for not registered custom module {point.module_scope}. NNCF only "
                    f"supports native PyTorch modules with respect to trainable parameter (weight) compressed, such "
                    f"as `torch.nn.Conv2d`. If your model contains a custom, non-PyTorch standard module with trainable"
                    f" weights that should be compressed, you can register it using the "
                    f"`@nncf.register_module` decorator. Please refer to `Compression of custom modules` section in "
                    f"docs/Usage.md for more details."
                )

            norm_target_scope = self._normalize_variable_recurrent_scope(point.module_scope)
            norm_nncf_scopes = []
            for scope_list_for_module in self.get_nncf_module_scopes():
                norm_nncf_scopes.extend([self._normalize_variable_recurrent_scope(x) for x in scope_list_for_module])
            assert norm_target_scope in norm_nncf_scopes  # Required for proper Recurrent/VariableRecurrent addressing
            if point.insertion_type == PTInsertionType.NNCF_MODULE_PRE_OP:
                handle = nncf_module.register_pre_forward_operation(fn)
            elif point.insertion_type == PTInsertionType.NNCF_MODULE_POST_OP:
                handle = nncf_module.register_post_forward_operation(fn)
        else:
            raise nncf.ValidationError("Unsupported insertion type: {}".format(point.insertion_type))
        self._groups_vs_hooks_handlers[hooks_group_name].append(handle)
        return handle

class NNCFNetwork(torch.nn.Module, metaclass=NNCFNetworkMeta):
    """
    A mixin-like class to dynamically extend the original model object's class with.
    """

    TRACE_PARAMETERS_KEY = "trace_parameters"

    def __init__(self, *args, **kwargs):
        """
        In normal situations, the __init__ of the NNCFNetwork will never be called. The constructor-like syntax is
        achieved by a __call__ method defined in the metaclass `NNCFNetworkMeta`.
        """
        super().__init__()
        raise nncf.InternalError("Direct instantiation of NNCFNetwork objects using __init__ is prohibited.")

    def __call__(self, *args, **kwargs):
        """
        Ensures that functor-like calls of the processed model object will directly trigger the NNCF-specific
        forward call.
        """
        return ORIGINAL_CALL(self, *args, **kwargs)

    def forward(self, *args, **kwargs):
        """
        Wraps the original forward call, doing additional actions before and after the call to facilitate model
        graph tracing and calling compression-related hooks.
        """

        with self.nncf._compressed_context as ctx:
            ctx.base_module_thread_local_replica = self

            # add tracing capabilities to model parameters
            if self.nncf.trace_parameters:
                wrap_parameters(self)

            args, kwargs = replicate_same_tensors((args, kwargs))
            if not self.nncf._in_user_dummy_forward:
                # If a user supplies own dummy forward, he is responsible for
                # correctly wrapping inputs inside it as well.
                args, kwargs = strip_traced_tensors(args, kwargs)
                args, kwargs = self.nncf._wrap_inputs_fn(args, kwargs)

            # For purposes of scope tracking, need the original forward call to occur as if it were
            # a module call of the corresponding object.
            if self.nncf._original_instance_forward is not None:

                def _unbound_like_original_instance_forward(_self, *args, **kwargs):
                    return self.nncf._original_instance_forward(*args, **kwargs)

                retval = wrap_module_call(_unbound_like_original_instance_forward)(self, *args, **kwargs)

            elif self.nncf._bound_original_forward is None:
                retval = wrap_module_call(self.nncf._original_unbound_forward)(self, *args, **kwargs)
            else:

                def _unbound_like_original_forward(_self, *args, **kwargs):
                    return self.nncf._bound_original_forward(*args, **kwargs)

                retval = wrap_module_call(_unbound_like_original_forward)(self, *args, **kwargs)

            retval = replicate_same_tensors(retval)
            if not self.nncf._in_user_dummy_forward:
                retval = self.nncf._wrap_outputs_fn(retval)

        if self.nncf._kd_loss_handler is not None and self.training:
            self.nncf._kd_loss_handler(retval, *args, **kwargs)
        return retval

    @property
    def nncf(self) -> NNCFNetworkInterface:
        """
        Accessor for all NNCF-specific methods and attributes of the compressed model object.
        """
        # self._nncf is being set in the creation function defined in the NNCFNetworkMeta metaclass
        return self._nncf

    def __setattr__(self, key, value):
        # If setting `forward`, set it on the original model.
        if key == "forward":
            nncf_logger.warning(
                "You are setting `forward` on an NNCF-processed model object.\n"
                "NNCF relies on custom-wrapping the `forward` call in order to function properly.\n"
                "Arbitrary adjustments to the forward function on an NNCFNetwork object have undefined behavior.\n"
                "If you need to replace the underlying forward function of the original model so that "
                "NNCF should be using that instead of the original forward function that NNCF saved "
                "during the compressed model creation, you can do this by calling:\n"
                "model.nncf.set_original_unbound_forward(fn)\n"
                "if `fn` has an unbound 0-th `self` argument, or\n"
                "with model.nncf.temporary_bound_original_forward(fn): ...\n"
                "if `fn` already had 0-th `self` argument bound or never had it in the first place."
            )
        super().__setattr__(key, value)

