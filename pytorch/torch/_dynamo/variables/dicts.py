class ConstDictVariable(VariableTracker):
    _nonvar_fields = {
        "user_cls",
        *VariableTracker._nonvar_fields,
    }

    class _HashableTracker:
        """
        Auxiliary opaque internal class that wraps a VariableTracker and makes it hashable
        This should not be seen or touched by anything outside of ConstDictVariable and its children
        Note that it's also fine to put VTs into dictionaries and sets, but doing so does not take into account aliasing
        """

        def __init__(self, vt):
            # We specialize SymNodes
            vt = specialize_symnode(vt)
            # TODO Temorarily remove to figure out what keys are we breaking on
            # and add proper support for them
            if not is_hashable(vt):
                unimplemented(f"Dict key of type {type(vt)}. Key: {vt}")
            self.vt = vt

        @property
        def underlying_value(self):
            if isinstance(self.vt, variables.TensorVariable):
                x = self.vt.as_proxy().node.meta["example_value"]
            elif isinstance(self.vt, variables.TupleVariable):
                Hashable = ConstDictVariable._HashableTracker
                x = tuple(Hashable(e).underlying_value for e in self.vt.items)
            elif isinstance(self.vt, variables.NNModuleVariable):
                return self.vt.module
            elif isinstance(self.vt, variables.UnspecializedNNModuleVariable):
                return self.vt.value
            elif isinstance(self.vt, variables.UserFunctionVariable):
                return self.vt.get_function()
            else:
                x = self.vt.as_python_constant()
            return x

        def __hash__(self):
            return hash(self.underlying_value)

        @staticmethod
        def _eq_impl(a, b):
            # TODO: Put this in utils and share it between variables/builtin.py and here
            if type(a) != type(b):
                return False
            elif isinstance(a, tuple):
                Hashable = ConstDictVariable._HashableTracker
                return len(a) == len(b) and all(
                    Hashable._eq_impl(u, v) for u, v in zip(a, b)
                )
            elif is_fake(a):
                return a is b
            else:
                return a == b

        def __eq__(self, other: "ConstDictVariable._HashableTracker") -> bool:
            Hashable = ConstDictVariable._HashableTracker
            assert isinstance(other, Hashable) or ConstantVariable.is_literal(
                other
            ), type(other)
            if isinstance(other, Hashable):
                return Hashable._eq_impl(self.underlying_value, other.underlying_value)

            # constant
            return Hashable._eq_impl(self.underlying_value, other)

    def __init__(
        self, items: Dict[VariableTracker, VariableTracker], user_cls=dict, **kwargs
    ):
        super().__init__(**kwargs)

        Hashable = ConstDictVariable._HashableTracker

        # Keys will just be HashableTrackers when cloning, in any other case they'll be VariableTrackers
        assert all(
            isinstance(x, (VariableTracker, Hashable))
            and isinstance(v, VariableTracker)
            for x, v in items.items()
        )

        def make_hashable(key):
            return key if isinstance(key, Hashable) else Hashable(key)

        self.items = {make_hashable(x): v for x, v in items.items()}
        self.user_cls = user_cls

    def as_proxy(self):
        return {k.vt.as_proxy(): v.as_proxy() for k, v in self.items.items()}

    def debug_repr(self):
        return (
            "{"
            + ", ".join(
                f"{k.vt.debug_repr()}: {v.debug_repr()}" for k, v in self.items.items()
            )
            + "}"
        )

    def as_python_constant(self):
        return {
            k.vt.as_python_constant(): v.as_python_constant()
            for k, v in self.items.items()
        }

    def keys_as_python_constant(self):
        return {k.vt.as_python_constant(): v for k, v in self.items.items()}

    def python_type(self):
        return self.user_cls

    def __contains__(self, vt):
        assert isinstance(vt, VariableTracker)
        Hashable = ConstDictVariable._HashableTracker
        return (
            is_hashable(vt)
            and Hashable(vt) in self.items
            and not isinstance(self.items[Hashable(vt)], variables.DeletedVariable)
        )

    def len(self):
        return len(
            [
                x
                for x in self.items.values()
                if not isinstance(x, variables.DeletedVariable)
            ]
        )

    def reconstruct(self, codegen):
        # instructions to load collections.OrderedDict if necessary
        if self.user_cls is collections.OrderedDict:
            codegen.add_push_null(
                lambda: codegen.extend_output(
                    [
                        codegen.create_load_python_module(collections),
                        codegen.create_load_attr("OrderedDict"),
                    ]
                )
            )
        # instructions to build the dict keys and values
        for key, value in self.items.items():
            codegen(key.vt)
            codegen(value)
        # BUILD_MAP and calling collections.OrderedDict if necessary
        if self.user_cls is collections.OrderedDict:
            codegen.extend_output(
                [
                    create_instruction("BUILD_MAP", arg=len(self.items)),
                    *create_call_function(1, False),
                ]
            )
        # BUILD_MAP only if user_cls is dict
        else:
            codegen.append_output(create_instruction("BUILD_MAP", arg=len(self.items)))

    def getitem_const(self, arg: VariableTracker):
        key = ConstDictVariable._HashableTracker(arg)
        if key not in self.items:
            unimplemented(f"dict KeyError: {arg.value}")
        return self.items[key]

    def maybe_getitem_const(self, arg: VariableTracker):
        key = ConstDictVariable._HashableTracker(arg)
        if key not in self.items:
            return None
        return self.items[key]

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from . import (
            BuiltinVariable,
            ConstantVariable,
            ListIteratorVariable,
            ListVariable,
            TupleVariable,
        )

        Hashable = ConstDictVariable._HashableTracker

        arg_hashable = args and is_hashable(args[0])

        if name == "__getitem__":
            assert len(args) == 1
            return self.getitem_const(args[0])
        elif name == "items":
            assert not (args or kwargs)
            if self.source:
                tx.output.guard_on_key_order.add(self.source.name())
            return TupleVariable(
                [TupleVariable([k.vt, v]) for k, v in self.items.items()]
            )
        elif name == "keys":
            if self.source:
                tx.output.guard_on_key_order.add(self.source.name())
            assert not (args or kwargs)
            return DictKeys(self)
        elif name == "values":
            if self.source:
                tx.output.guard_on_key_order.add(self.source.name())
            assert not (args or kwargs)
            return DictValues(self)
        elif name == "copy":
            assert not (args or kwargs)
            return self.clone(items=self.items.copy(), mutable_local=MutableLocal())
        elif name == "__len__":
            assert not (args or kwargs)
            return ConstantVariable.create(len(self.items))
        elif name == "__setitem__" and arg_hashable and self.mutable_local:
            assert not kwargs and len(args) == 2
            tx.output.side_effects.mutation(self)
            self.items[Hashable(args[0])] = args[1]
            return ConstantVariable.create(None)
        elif name == "__delitem__" and arg_hashable and self.mutable_local:
            tx.output.side_effects.mutation(self)
            self.items.__delitem__(Hashable(args[0]))
            return ConstantVariable.create(None)
        elif name in ("pop", "get") and len(args) in (1, 2) and args[0] not in self:
            # missing item, return the default value
            if len(args) == 1:
                return ConstantVariable(None)
            else:
                return args[1]
        elif name == "pop" and arg_hashable and self.mutable_local:
            tx.output.side_effects.mutation(self)
            return self.items.pop(Hashable(args[0]))
        elif name == "clear":
            tx.output.side_effects.mutation(self)
            self.items.clear()
            return ConstantVariable.create(None)
        elif (
            name == "update"
            and len(args) == 1
            and isinstance(
                args[0],
                (
                    ConstDictVariable,
                    ListVariable,
                    TupleVariable,
                    ListIteratorVariable,
                ),
            )
            and self.mutable_local
        ):
            tx.output.side_effects.mutation(self)
            if isinstance(args[0], ConstDictVariable):
                dict_vt = args[0]
            else:
                dict_vt = BuiltinVariable.call_custom_dict(tx, dict, args[0])
            self.items.update(dict_vt.items)
            # Wrap strings
            kwargs = {
                Hashable(ConstantVariable.create(k)): v for k, v in kwargs.items()
            }
            self.items.update(kwargs)
            return ConstantVariable.create(None)
        elif name in ("get", "__getattr__") and args[0] in self:
            return self.getitem_const(args[0])
        elif name == "__contains__" and len(args) == 1:
            return ConstantVariable.create(args[0] in self)
        else:
            return super().call_method(tx, name, args, kwargs)

    def unpack_var_sequence(self, tx):
        return [x.vt for x in self.items.keys()]

