def add_generated_native_functions(
    rs: list[NativeFunction],
    indices: dict[DispatchKey, dict[OperatorName, BackendMetadata]],
) -> None:
    # The main code for generating new NativeFunctions
    # First we group of NativeFunctions by schema kind,
    # then we detect which ones are missing and generate them.
    pre_grouped_native_functions = pre_group_native_functions(rs)
    for d in pre_grouped_native_functions.values():
        has_functional = SchemaKind.functional in d
        has_inplace = SchemaKind.inplace in d
        has_mutable = SchemaKind.mutable in d
        has_out = SchemaKind.out in d

        # We automatically generate a few native functions that don't exist in the yaml, for a few reasons:
        # (1) If an operator has an inplace/out= variant but no functional variant, we can generate
        #     a simple functional variant that the functionalization pass can consume.
        # (2) If an operator has an inplace or functional but no out= variant, we generate an out=
        #     variant, mostly so we can easily pair up functions into NativeFunctionsGroup,
        #     while maintaining the constraint that the out= variant is "required".
        if has_mutable or has_inplace or has_out or has_functional:
            # Don't bother generating functions trio's for native functions that bypass the dispatcher.
            are_manual = all(f.manual_cpp_binding for f in d.values())
            # Don't bother generating functional + out= variants for view operators
            # set_ is technically an inplace_view, but for now it is treated
            # as a normal inplace op in the codegen
            has_view_ops = any(
                f.is_view_op and str(f.func.name.name) != "set_" for f in d.values()
            )
            # Don't generate the other variants for CompositeImplicitAutograd operators.
            # We could probably do this, but the main benefit of generating the function triplets
            # is for transforms that need them, and transforms don't need to act directly
            # on CompositeImplicitAutograd operators (since we let them decompose).
            are_composite_implicit = all(
                f.has_composite_implicit_autograd_kernel for f in d.values()
            )
            if are_manual or has_view_ops or are_composite_implicit:
                continue
            if has_out and len(d.values()) == 1:
                # Note: [Out ops with functional variants that don't get grouped properly]
                # In theory we could validly have an out= operator in native_functions.yaml
                # that has no other variants.
                # But today, all of the operators where that's the case actually do have
                # functional variants, that we are just unable to pair up properly.
                # I think banning this all together is probably safer
                # (you can always add a functional variant yourself if you want to add a new out= operator).
                #
                # We should probably fix the existing cases; this check is to prevent us from adding more over time.
                if (
                    str(d[SchemaKind.out].func.name)
                    not in OUT_OPS_THAT_DONT_GET_GROUPED_PROPERLY
                ):
                    raise AssertionError(
                        f"Found an out= operator that we could not find any other variants of: {str(d[SchemaKind.out].func)}"
                    )
                continue

            # Some inplace ops that have problematic schemas (that we should fix), which prevent us
            # from generating out= and functional variants
            if (
                has_inplace
                and str(d[SchemaKind.inplace].func.name)
                in INPLACE_OPS_THAT_DONT_GET_GROUPED_PROPERLY
            ):
                continue

            base_fn = (
                d[SchemaKind.inplace]
                if has_inplace
                else d[SchemaKind.mutable]
                if has_mutable
                else d[SchemaKind.out]
                if has_out
                else d[SchemaKind.functional]
            )

            # Note: [Mutable ops that cannot get an out variant]
            # We can only generate an out= variant if either:
            # - the original function has tensor-like returns (since we can convert them to out kwargs)
            # - or it's inplace (since we can convert `self` to an out kwarg)
            # There are only two functions that don't fit this criteria today though,
            # and they both look like they should be fixed to be out= variants,
            # so if feels safer to ban this schema all-together
            base_fn_valid = base_fn.func.kind() == SchemaKind.inplace or any(
                r.type.is_tensor_like() for r in base_fn.func.returns
            )
            # Note: [Loosen the assertion that all functional should have out variant]
            # By design all functional operators should have our variants. The needs_out check
            # is loosening this requirement, changing it to only generate out variant if there's
            # an `autogen` block in the native function, in the long run it should be removed.
            # FIXME: Remove this after figuring out CI job failures related to min, max, mean
            needs_out = any("out" in str(op_name) for op_name in base_fn.autogen)
            gets_out_variant = not has_out and base_fn_valid and needs_out
            if not has_out and not base_fn_valid:
                if (
                    str(base_fn.func.name)
                    not in MUTABLE_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT
                    and str(base_fn.func.name)
                    not in FUNCTIONAL_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT
                ):
                    raise AssertionError(
                        f"""Found an operator that we could not generate an out= variant for: {str(base_fn.func)}.
This type of operators don't have tensor-like return, making it difficult to generate a proper out= variant. If
out= variant is not needed, please add the function name into FUNCTIONAL_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT list."""
                    )

            # Generate an out= variant
            if gets_out_variant:
                fn, metadata = generate_function(base_fn, SchemaKind.out)
                d[SchemaKind.out] = fn
                BackendIndex.grow_index(indices, metadata)
                rs.append(fn)

            # Generate a functional variant, but only do it if the operator got an out= variant
            # (Functional variants are only useful if we can group up the variants,
            # which we can only do if they have an out= variant)
            if not has_functional and (has_out or gets_out_variant):
                fn, metadata = generate_function(base_fn, SchemaKind.functional)
                d[SchemaKind.functional] = fn
                BackendIndex.grow_index(indices, metadata)
                rs.append(fn)