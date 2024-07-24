def translate_native_yaml(
    tags_yaml_path: str,
    aten_yaml_path: str,
    native_yaml_path: str | None,
    use_aten_lib: bool,
    out_file: TextIO,
) -> None:
    """Translates Executorch DSL dialect to use the same syntax as
    native_functions.yaml. The major difference is that Executorch DSL dialect
    supports "op" key, where it refers to the operator name in native_functions.yaml.

    For example, a functions.yaml may have the following entry:

    - op: add.out
      ...

    It needs to be translated to the following:

    - func: add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
      ...

    We go in aten_yaml_path and find the operator schema for "add.out" and add it
    to the original functions.yaml. We also add required field "variants", where for
    Executorch it will always be "function".

    For ATen mode we don't have to do the translation because native_yaml_path is
    the same as native_functions.yaml.

    Args:
        tags_yaml_path: Path to a tags.yaml file to satisfy codegen parsing.
            It is not optional.
        aten_yaml_path: Path to ATen operator yaml file native_functions.yaml.
        native_yaml_path: Path to a functions.yaml file to parse.
            If the path does not exist in the filesystem, it is treated as an
            empty file. If `custom_ops_yaml_path` exists, the contents of that
            file are appended to the yaml input to be parsed.
        use_aten_lib: We use this flag to determine if we want to generate native
            functions. In ATen mode we should generate out= variants.
        out_file: The IO object that we are writing into.
    Returns:
        None
    """
    if use_aten_lib:
        with open(aten_yaml_path) as aten_yaml:
            out_file.writelines(aten_yaml.readlines())
        return

    native_functions, persisted_fields = parse_et_yaml(
        aten_yaml_path,
        tags_yaml_path,
        None,
        skip_native_fns_gen=False,
    )

    func_to_scoped_name: dict[FunctionSchema, str] = {
        f.func: f"{f.namespace}::{f.func.name}" for f in native_functions
    }
    op_to_scoped_name: dict[OperatorName, str] = {
        func.name: name for func, name in func_to_scoped_name.items()
    }

    schema_dict = {name: str(func) for func, name in func_to_scoped_name.items()}
    kernel_persist_dict: dict[str, dict[str, Any]] = {
        op_to_scoped_name[op]: v for op, v in persisted_fields.items()
    }

    if (
        not native_yaml_path
        or not os.path.exists(native_yaml_path)
        or os.stat(native_yaml_path).st_size == 0
    ):
        return
    with open(native_yaml_path) as native_yaml:
        native_es = yaml.load(native_yaml, Loader=LineLoader)
        if not native_es:
            return
        for e in native_es:
            assert isinstance(e.get("__line__"), int), e
            loc = Location(native_yaml_path, e.pop("__line__"))
            with context(lambda: f"in {loc}:\n  "):
                if "variants" not in e:
                    e["variants"] = "function"
                if "func" in e:
                    continue
                assert isinstance(e.get("op"), str), e
                opname = e.pop("op")
                if "::" not in opname:
                    opname = "aten::" + opname
                assert opname in schema_dict
                e["func"] = schema_dict.get(opname)

                # Write out persisted kernel information
                if opname in kernel_persist_dict:
                    for k, v in kernel_persist_dict[opname].items():
                        e[k] = v

        yaml.dump(native_es, out_file, width=1000)

class ComputeCodegenUnboxedKernels:
    selector: SelectiveBuilder

    use_aten_lib: bool

    @method_with_nested_native_function
    def __call__(
        self,
        unbox_kernel_entry: tuple[NativeFunction, tuple[ETKernelKey, BackendMetadata]],
    ) -> str:
        f: NativeFunction = unbox_kernel_entry[0]
        kernel_key: ETKernelKey | list[ETKernelKey] = unbox_kernel_entry[1][0]
        kernel_meta: BackendMetadata = unbox_kernel_entry[1][1]

        op_name = f"{f.namespace}::{f.func.name}"
        if not self.selector.is_root_operator(op_name):
            return ""

        if not isinstance(kernel_key, list):
            kernel_key = [kernel_key]
        used_kernel_keys = self.selector.et_get_selected_kernels(
            op_name, [k.to_native_string() for k in kernel_key]
        )
        if not used_kernel_keys:
            return ""
        sig: CppSignature | ExecutorchCppSignature
        argument_type_gen: Callable[..., NamedCType]
        return_type_gen: Callable[..., CType]
        if self.use_aten_lib:
            sig = CppSignatureGroup.from_native_function(
                f, method=False, fallback_binding=f.manual_cpp_binding
            ).most_faithful_signature()
            argument_type_gen = aten_cpp.argumenttype_type
            return_type_gen = aten_cpp.returns_type
            arguments = sig.arguments()
            kernel_call = f"torch::executor::{f.namespace}::{sig.name()}"
        else:
            sig = ExecutorchCppSignature.from_native_function(f)
            argument_type_gen = et_cpp.argumenttype_type
            return_type_gen = et_cpp.returns_type
            arguments = sig.arguments(include_context=False)
            kernel_call = f"{kernel_meta.cpp_namespace}::{kernel_meta.kernel}"
        # parse arguments into C++ code
        binding_list, code_list = Unboxing(
            argument_type_gen=argument_type_gen
        ).convert_arguments(arguments)

        # for each C++ argument, generate the conversion code
        code_connector = "\n\t"
        arg_connector = ", "

        args_str = f"{arg_connector.join(e.name for e in binding_list)}"
        event_tracer_output_logging = ""
        output_ids = []

        if len(f.func.returns) == 0:
            if len(f.func.arguments.out) == 0:
                raise Exception(  # noqa: TRY002
                    f"Can't handle native function {f.func} with no returns and no out yet."
                )
            out = f.func.arguments.out[0]
            return_assignment = f"""stack[{len(binding_list)}] = &{out.name};"""
            ret_prefix = ""
            output_ids = [len(binding_list)]
        else:
            if len(f.func.arguments.out) == 0:
                return_assignment = (
                    f"""*stack[{len(binding_list)}] = EValue(result_);"""
                )
                ret_prefix = return_type_gen(f.func.returns).cpp_type() + " result_ = "
                output_ids = [len(binding_list)]
            else:
                return_assignment = ""
                ret_prefix = ""
                output_ids = [
                    len(binding_list) - (i + 1)
                    for i in reversed(range(len(f.func.arguments.out)))
                ]

        for output_id in output_ids:
            event_tracer_output_logging += (
                f"internal::event_tracer_log_evalue("
                f"context.internal_event_tracer(), "
                f"*stack[{output_id}]);\n"
            )

        newline = "\n    "
        return "\n".join(
            [
                f"""
Kernel(
    "{f.namespace}::{f.func.name}",{newline + '"' + (k + '",') if k != 'default' else ''}
    []({contextArg.defn()}, EValue** stack) {{
        {code_connector.join(code_list)}

        internal::EventTracerProfileScope event_tracer_scope(context.internal_event_tracer(), "native_call_{f.func.name}");
        EXECUTORCH_SCOPE_PROF("native_call_{f.func.name}");
        {ret_prefix}{kernel_call}(context, {args_str});
        {event_tracer_output_logging}
        {return_assignment}
    }}
),
"""
                for k in used_kernel_keys
            ]
        )

