def run(
    source_yaml: str, output_dir: str, dry_run: bool, impl_path: str | None = None
) -> None:
    # Assumes that this file lives at PYTORCH_ROOT/torchgen/gen_backend_stubs.py
    pytorch_root = Path(__file__).parent.parent.absolute()
    template_dir = os.path.join(pytorch_root, "aten/src/ATen/templates")

    def make_file_manager(install_dir: str) -> FileManager:
        return FileManager(
            install_dir=install_dir, template_dir=template_dir, dry_run=dry_run
        )

    fm = make_file_manager(output_dir)

    native_yaml_path = os.path.join(
        pytorch_root, "aten/src/ATen/native/native_functions.yaml"
    )
    tags_yaml_path = os.path.join(pytorch_root, "aten/src/ATen/native/tags.yaml")
    parsed_yaml = parse_native_yaml(native_yaml_path, tags_yaml_path)
    native_functions, backend_indices = (
        parsed_yaml.native_functions,
        parsed_yaml.backend_indices,
    )
    grouped_native_functions = get_grouped_native_functions(native_functions)
    parsed_backend_yaml = parse_backend_yaml(
        source_yaml, grouped_native_functions, backend_indices
    )
    backend_key = parsed_backend_yaml.backend_key
    autograd_key = parsed_backend_yaml.autograd_key
    cpp_namespace = parsed_backend_yaml.cpp_namespace
    class_name = parsed_backend_yaml.class_name
    backend_indices = parsed_backend_yaml.backend_indices

    selector = SelectiveBuilder.get_nop_selector()

    if backend_key is None:
        # This could be useful if a backend wants to quickly set up a noop yaml file but doesn't have any kernels ready yet.
        return

    if class_name is None:
        # class_name is an optional argument to backend yaml file.
        # if specified it allows an external backend to override
        # the name of the class that all generated kernel definitions live under.
        # if not specified, its value is given as native_function_class_name.
        class_name = backend_indices[backend_key].native_function_class_name()
    assert class_name is not None

    if impl_path is not None:
        error_on_missing_kernels(
            native_functions,
            backend_indices,
            backend_key,
            autograd_key,
            class_name,
            impl_path,
        )

    gen_dispatchkey_nativefunc_headers(
        fm,
        class_name,
        cpp_namespace,
        backend_indices,
        grouped_native_functions,
        backend_key,
        autograd_key,
    )

    for dispatch_key in (
        [backend_key] if autograd_key is None else [backend_key, autograd_key]
    ):
        gen_dispatcher_registrations(
            fm,
            output_dir,
            class_name,
            backend_indices,
            grouped_native_functions,
            backend_key,
            dispatch_key,
            selector,
        )

