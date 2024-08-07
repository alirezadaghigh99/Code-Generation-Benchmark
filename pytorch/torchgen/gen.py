def get_native_function_schema_registrations(
    *,
    native_functions: Sequence[NativeFunction],
    schema_selector: SelectiveBuilder,
) -> tuple[list[str], str]:
    ns_native_functions: dict[str, list[NativeFunction]] = defaultdict(list)
    for native_function in native_functions:
        ns_native_functions[native_function.namespace].append(native_function)
    schema_registrations = ""
    aten_schema_registrations = []
    custom_namespace = None
    for namespace, funcs in ns_native_functions.items():
        schema_registrations_body = list(
            mapMaybe(RegisterSchema(schema_selector), funcs)
        )
        # NB: we have to separate aten namespace registration from other namespaces,
        # because in the template we hardcoded an operator for ATen already.
        if namespace == "aten":
            aten_schema_registrations = schema_registrations_body
        else:
            custom_namespace = namespace
            tab = "\t"
            # if the namespace is predefined, we should use define a library fragment
            # instead of a new library
            torch_library_macro = (
                "TORCH_LIBRARY_FRAGMENT"
                if namespace in FRAGMENT_NAMESPACES
                else "TORCH_LIBRARY"
            )
            schema_registrations += f"""
{torch_library_macro}({custom_namespace}, m) {{
  {tab.join(schema_registrations_body)}
}};"""
    return (aten_schema_registrations, schema_registrations)

def get_native_function_declarations(
    *,
    grouped_native_functions: Sequence[NativeFunction | NativeFunctionsGroup],
    backend_indices: dict[DispatchKey, BackendIndex],
    native_function_decl_gen: Callable[
        [NativeFunctionsGroup | NativeFunction, BackendIndex], list[str]
    ] = dest.compute_native_function_declaration,
) -> list[str]:
    """
    Generate kernel declarations, in `NativeFunction(s).h`.
    :param grouped_native_functions: a sequence of `NativeFunction` or `NativeFunctionGroup`.
    :param backend_indices: kernel collections grouped by dispatch key.
    :param native_function_decl_gen: callable to generate kernel declaration for each `NativeFunction`.
    :return: a list of string, from the string with all declarations, grouped by namespaces, split by newline.
    """

    ns_grouped_kernels = get_ns_grouped_kernels(
        grouped_native_functions=grouped_native_functions,
        backend_indices=backend_indices,
        native_function_decl_gen=native_function_decl_gen,
    )
    return get_native_function_declarations_from_ns_grouped_kernels(
        ns_grouped_kernels=ns_grouped_kernels
    )

