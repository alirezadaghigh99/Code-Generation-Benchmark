def main(args: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Generate unboxing source files")
    parser.add_argument(
        "-s",
        "--source-path",
        help="path to source directory for ATen",
        default="aten/src/ATen",
    )
    parser.add_argument(
        "-d",
        "--install-dir",
        "--install_dir",
        help="output directory",
        default="build/aten/src/ATen",
    )
    parser.add_argument(
        "-o",
        "--output-dependencies",
        help="output a list of dependencies into the given file and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run without writing any files (still updates outputs)",
    )
    parser.add_argument(
        "--op-selection-yaml-path",
        "--op_selection_yaml_path",
        help="Provide a path to the operator selection (for custom build) YAML "
        "that contains the information about the set of selected operators "
        "and their categories (training, ...). Each operator is either a "
        "full operator name with overload or just a bare operator name. "
        "The operator names also contain the namespace prefix (e.g. aten::)",
    )
    parser.add_argument(
        "--op-registration-allowlist",
        "--op_registration_allowlist",
        nargs="*",
        help="filter op registrations by the allowlist (if set); "
        "each item is `namespace`::`operator name` without overload name; "
        "e.g.: aten::empty aten::conv2d ...",
    )
    parser.add_argument(
        "--TEST-ONLY-op-registration-allowlist-yaml-path",
        "--TEST_ONLY_op_registration_allowlist_yaml_path",
        help="Provide a path to the operator selection (for custom build) YAML "
        "which contains a list of operators. It is to serve testing purpose and "
        "each item is `namespace`::`operator name` without overload name; "
        "e.g.: aten::empty aten::conv2d ...",
    )

    options = parser.parse_args(args)
    if options.op_registration_allowlist:
        op_registration_allowlist = options.op_registration_allowlist
    elif options.TEST_ONLY_op_registration_allowlist_yaml_path:
        with open(options.TEST_ONLY_op_registration_allowlist_yaml_path) as f:
            op_registration_allowlist = yaml.safe_load(f)
    else:
        op_registration_allowlist = None

    selector = get_custom_build_selector(
        op_registration_allowlist,
        options.op_selection_yaml_path,
    )

    native_yaml_path = os.path.join(options.source_path, "native/native_functions.yaml")
    tags_yaml_path = os.path.join(options.source_path, "native/tags.yaml")
    parsed_yaml = parse_native_yaml(native_yaml_path, tags_yaml_path)
    native_functions, backend_indices = (
        parsed_yaml.native_functions,
        parsed_yaml.backend_indices,
    )

    cpu_fm = make_file_manager(options=options)
    gen_unboxing(native_functions=native_functions, cpu_fm=cpu_fm, selector=selector)

    if options.output_dependencies:
        depfile_path = Path(options.output_dependencies).resolve()
        depfile_name = depfile_path.name
        depfile_stem = depfile_path.stem

        path = depfile_path.parent / depfile_name
        cpu_fm.write_outputs(depfile_stem, str(path))

