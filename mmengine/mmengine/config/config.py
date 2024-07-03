    def fromfile(filename: Union[str, Path],
                 use_predefined_variables: bool = True,
                 import_custom_modules: bool = True,
                 use_environment_variables: bool = True,
                 lazy_import: Optional[bool] = None,
                 format_python_code: bool = True) -> 'Config':
        """Build a Config instance from config file.

        Args:
            filename (str or Path): Name of config file.
            use_predefined_variables (bool, optional): Whether to use
                predefined variables. Defaults to True.
            import_custom_modules (bool, optional): Whether to support
                importing custom modules in config. Defaults to None.
            use_environment_variables (bool, optional): Whether to use
                environment variables. Defaults to True.
            lazy_import (bool): Whether to load config in `lazy_import` mode.
                If it is `None`, it will be deduced by the content of the
                config file. Defaults to None.
            format_python_code (bool): Whether to format Python code by yapf.
                Defaults to True.

        Returns:
            Config: Config instance built from config file.
        """
        filename = str(filename) if isinstance(filename, Path) else filename
        if lazy_import is False or \
           lazy_import is None and not Config._is_lazy_import(filename):
            cfg_dict, cfg_text, env_variables = Config._file2dict(
                filename, use_predefined_variables, use_environment_variables,
                lazy_import)
            if import_custom_modules and cfg_dict.get('custom_imports', None):
                try:
                    import_modules_from_strings(**cfg_dict['custom_imports'])
                except ImportError as e:
                    err_msg = (
                        'Failed to import custom modules from '
                        f"{cfg_dict['custom_imports']}, the current sys.path "
                        'is: ')
                    for p in sys.path:
                        err_msg += f'\n    {p}'
                    err_msg += (
                        '\nYou should set `PYTHONPATH` to make `sys.path` '
                        'include the directory which contains your custom '
                        'module')
                    raise ImportError(err_msg) from e
            return Config(
                cfg_dict,
                cfg_text=cfg_text,
                filename=filename,
                env_variables=env_variables,
            )
        else:
            # Enable lazy import when parsing the config.
            # Using try-except to make sure ``ConfigDict.lazy`` will be reset
            # to False. See more details about lazy in the docstring of
            # ConfigDict
            ConfigDict.lazy = True
            try:
                cfg_dict, imported_names = Config._parse_lazy_import(filename)
            except Exception as e:
                raise e
            finally:
                # disable lazy import to get the real type. See more details
                # about lazy in the docstring of ConfigDict
                ConfigDict.lazy = False

            cfg = Config(
                cfg_dict,
                filename=filename,
                format_python_code=format_python_code)
            object.__setattr__(cfg, '_imported_names', imported_names)
            return cfg    def _file2dict(
            filename: str,
            use_predefined_variables: bool = True,
            use_environment_variables: bool = True,
            lazy_import: Optional[bool] = None) -> Tuple[dict, str, dict]:
        """Transform file to variables dictionary.

        Args:
            filename (str): Name of config file.
            use_predefined_variables (bool, optional): Whether to use
                predefined variables. Defaults to True.
            use_environment_variables (bool, optional): Whether to use
                environment variables. Defaults to True.
            lazy_import (bool): Whether to load config in `lazy_import` mode.
                If it is `None`, it will be deduced by the content of the
                config file. Defaults to None.

        Returns:
            Tuple[dict, str]: Variables dictionary and text of Config.
        """
        if lazy_import is None and Config._is_lazy_import(filename):
            raise RuntimeError(
                'The configuration file type in the inheritance chain '
                'must match the current configuration file type, either '
                '"lazy_import" or non-"lazy_import". You got this error '
                'since you use the syntax like `with read_base(): ...` '
                f'or import non-builtin module in {filename}. See more '
                'information in https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html'  # noqa: E501
            )

        filename = osp.abspath(osp.expanduser(filename))
        check_file_exist(filename)
        fileExtname = osp.splitext(filename)[1]
        if fileExtname not in ['.py', '.json', '.yaml', '.yml']:
            raise OSError('Only py/yml/yaml/json type are supported now!')
        try:
            with tempfile.TemporaryDirectory() as temp_config_dir:
                temp_config_file = tempfile.NamedTemporaryFile(
                    dir=temp_config_dir, suffix=fileExtname, delete=False)
                if platform.system() == 'Windows':
                    temp_config_file.close()

                # Substitute predefined variables
                if use_predefined_variables:
                    Config._substitute_predefined_vars(filename,
                                                       temp_config_file.name)
                else:
                    shutil.copyfile(filename, temp_config_file.name)
                # Substitute environment variables
                env_variables = dict()
                if use_environment_variables:
                    env_variables = Config._substitute_env_variables(
                        temp_config_file.name, temp_config_file.name)
                # Substitute base variables from placeholders to strings
                base_var_dict = Config._pre_substitute_base_vars(
                    temp_config_file.name, temp_config_file.name)

                # Handle base files
                base_cfg_dict = ConfigDict()
                cfg_text_list = list()
                for base_cfg_path in Config._get_base_files(
                        temp_config_file.name):
                    base_cfg_path, scope = Config._get_cfg_path(
                        base_cfg_path, filename)
                    _cfg_dict, _cfg_text, _env_variables = Config._file2dict(
                        filename=base_cfg_path,
                        use_predefined_variables=use_predefined_variables,
                        use_environment_variables=use_environment_variables,
                        lazy_import=lazy_import,
                    )
                    cfg_text_list.append(_cfg_text)
                    env_variables.update(_env_variables)
                    duplicate_keys = base_cfg_dict.keys() & _cfg_dict.keys()
                    if len(duplicate_keys) > 0:
                        raise KeyError(
                            'Duplicate key is not allowed among bases. '
                            f'Duplicate keys: {duplicate_keys}')

                    # _dict_to_config_dict will do the following things:
                    # 1. Recursively converts ``dict`` to :obj:`ConfigDict`.
                    # 2. Set `_scope_` for the outer dict variable for the base
                    # config.
                    # 3. Set `scope` attribute for each base variable.
                    # Different from `_scope_`, `scope` is not a key of base
                    # dict, `scope` attribute will be parsed to key `_scope_`
                    # by function `_parse_scope` only if the base variable is
                    # accessed by the current config.
                    _cfg_dict = Config._dict_to_config_dict(_cfg_dict, scope)
                    base_cfg_dict.update(_cfg_dict)

                if filename.endswith('.py'):
                    with open(temp_config_file.name, encoding='utf-8') as f:
                        parsed_codes = ast.parse(f.read())
                        parsed_codes = RemoveAssignFromAST(BASE_KEY).visit(
                            parsed_codes)
                    codeobj = compile(parsed_codes, filename, mode='exec')
                    # Support load global variable in nested function of the
                    # config.
                    global_locals_var = {BASE_KEY: base_cfg_dict}
                    ori_keys = set(global_locals_var.keys())
                    eval(codeobj, global_locals_var, global_locals_var)
                    cfg_dict = {
                        key: value
                        for key, value in global_locals_var.items()
                        if (key not in ori_keys and not key.startswith('__'))
                    }
                elif filename.endswith(('.yml', '.yaml', '.json')):
                    cfg_dict = load(temp_config_file.name)
                # close temp file
                for key, value in list(cfg_dict.items()):
                    if isinstance(value,
                                  (types.FunctionType, types.ModuleType)):
                        cfg_dict.pop(key)
                temp_config_file.close()

                # If the current config accesses a base variable of base
                # configs, The ``scope`` attribute of corresponding variable
                # will be converted to the `_scope_`.
                Config._parse_scope(cfg_dict)
        except Exception as e:
            if osp.exists(temp_config_dir):
                shutil.rmtree(temp_config_dir)
            raise e

        # check deprecation information
        if DEPRECATION_KEY in cfg_dict:
            deprecation_info = cfg_dict.pop(DEPRECATION_KEY)
            warning_msg = f'The config file {filename} will be deprecated ' \
                'in the future.'
            if 'expected' in deprecation_info:
                warning_msg += f' Please use {deprecation_info["expected"]} ' \
                    'instead.'
            if 'reference' in deprecation_info:
                warning_msg += ' More information can be found at ' \
                    f'{deprecation_info["reference"]}'
            warnings.warn(warning_msg, DeprecationWarning)

        cfg_text = filename + '\n'
        with open(filename, encoding='utf-8') as f:
            # Setting encoding explicitly to resolve coding issue on windows
            cfg_text += f.read()

        # Substitute base variables from strings to their actual values
        cfg_dict = Config._substitute_base_vars(cfg_dict, base_var_dict,
                                                base_cfg_dict)
        cfg_dict.pop(BASE_KEY, None)

        cfg_dict = Config._merge_a_into_b(cfg_dict, base_cfg_dict)
        cfg_dict = {
            k: v
            for k, v in cfg_dict.items() if not k.startswith('__')
        }

        # merge cfg_text
        cfg_text_list.append(cfg_text)
        cfg_text = '\n'.join(cfg_text_list)

        return cfg_dict, cfg_text, env_variables    def fromfile(filename: Union[str, Path],
                 use_predefined_variables: bool = True,
                 import_custom_modules: bool = True,
                 use_environment_variables: bool = True,
                 lazy_import: Optional[bool] = None,
                 format_python_code: bool = True) -> 'Config':
        """Build a Config instance from config file.

        Args:
            filename (str or Path): Name of config file.
            use_predefined_variables (bool, optional): Whether to use
                predefined variables. Defaults to True.
            import_custom_modules (bool, optional): Whether to support
                importing custom modules in config. Defaults to None.
            use_environment_variables (bool, optional): Whether to use
                environment variables. Defaults to True.
            lazy_import (bool): Whether to load config in `lazy_import` mode.
                If it is `None`, it will be deduced by the content of the
                config file. Defaults to None.
            format_python_code (bool): Whether to format Python code by yapf.
                Defaults to True.

        Returns:
            Config: Config instance built from config file.
        """
        filename = str(filename) if isinstance(filename, Path) else filename
        if lazy_import is False or \
           lazy_import is None and not Config._is_lazy_import(filename):
            cfg_dict, cfg_text, env_variables = Config._file2dict(
                filename, use_predefined_variables, use_environment_variables,
                lazy_import)
            if import_custom_modules and cfg_dict.get('custom_imports', None):
                try:
                    import_modules_from_strings(**cfg_dict['custom_imports'])
                except ImportError as e:
                    err_msg = (
                        'Failed to import custom modules from '
                        f"{cfg_dict['custom_imports']}, the current sys.path "
                        'is: ')
                    for p in sys.path:
                        err_msg += f'\n    {p}'
                    err_msg += (
                        '\nYou should set `PYTHONPATH` to make `sys.path` '
                        'include the directory which contains your custom '
                        'module')
                    raise ImportError(err_msg) from e
            return Config(
                cfg_dict,
                cfg_text=cfg_text,
                filename=filename,
                env_variables=env_variables,
            )
        else:
            # Enable lazy import when parsing the config.
            # Using try-except to make sure ``ConfigDict.lazy`` will be reset
            # to False. See more details about lazy in the docstring of
            # ConfigDict
            ConfigDict.lazy = True
            try:
                cfg_dict, imported_names = Config._parse_lazy_import(filename)
            except Exception as e:
                raise e
            finally:
                # disable lazy import to get the real type. See more details
                # about lazy in the docstring of ConfigDict
                ConfigDict.lazy = False

            cfg = Config(
                cfg_dict,
                filename=filename,
                format_python_code=format_python_code)
            object.__setattr__(cfg, '_imported_names', imported_names)
            return cfg