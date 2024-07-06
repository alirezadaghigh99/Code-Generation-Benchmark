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

