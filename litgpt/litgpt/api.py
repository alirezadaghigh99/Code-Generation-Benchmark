def load(
        cls,
        model: str,
        accelerator: Literal["cpu", "cuda", "auto"] = "auto",
        devices: Union[int, List[int]] = 1,
        quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8"]] = None,
        precision: Optional[Any] = None,
        init: Optional[Literal["pretrained", "random"]] = "pretrained",
        tokenizer_dir: Optional[Path] = None,
        access_token: Optional[str] = None
    ) -> "LLM":
        """
        Loads the LLM from a local directory or model hub.

        Arguments
            model: A local path to a directory containing the model weights or a valid model name.
               You can get a list of valid model names via the `litgpt download list` command line argument.
            accelerator: Which device type to load the model on ("cpu", "gpu", "mps", "cuda", or "auto")
            devices: The number of devices (1, 2, etc.) or device IDs (e.g., [0, 2] to use the first and third GPU).
            quantize: Whether to quantize the model and using which method:
                - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
                - bnb.int8: 8-bit quantization from bitsandbytes
                for more details, see https://github.com/Lightning-AI/litgpt/blob/main/tutorials/quantize.md
            precision: Indicates the Fabric precision setting to use.
                For instance, "32-true", "16-mixed", "16-true", "bf16-mixed", "bf16-true".
                For more details, see https://lightning.ai/docs/fabric/stable/api/fabric_args.html#precision
            init: If "pretrained" (default), downloads the model from the HF Hub if a local model can't be found at the `model`
                directory name; otherwise loads the model from the local directory.
                If "random", initializes the `model` with random weights.
            access_token:
                Optional API token to access models with restrictions when using `init="pretrained"`.
            tokenizer_dir: An optional tokenizer directory if `model` is not a checkpoint directory, or if a user
                wants to use a different tokenizer instead.
        """
        allowed_accelerators = {"cpu", "gpu", "cuda", "mps", "auto"}
        if accelerator not in allowed_accelerators:
            raise ValueError(f"Invalid accelerator: {accelerator}. Must be one of {allowed_accelerators}.")

        if accelerator == "auto":
            if torch.cuda.is_available():
                accelerator = "cuda"
            elif torch.backends.mps.is_available():
                accelerator = "mps"
            else:
                accelerator = "cpu"

        num_devices = calculate_number_of_devices(devices)

        if num_devices > 1:
            raise NotImplementedError(
                "Support for multiple devices is currently not implemented, yet."
            )

        allowed_init = {"pretrained", "random"}

        if init == "pretrained":
            from litgpt.scripts.download import download_from_hub  # Moved here due to the circular import issue in LitGPT that we need to solve some time

            checkpoint_dir = extend_checkpoint_dir(Path(model))
            try:
                check_valid_checkpoint_dir(checkpoint_dir, verbose=False, raise_error=True)
            except FileNotFoundError:
                if not access_token:
                    access_token = os.getenv("HF_TOKEN")
                download_from_hub(repo_id=model, access_token=access_token)

            checkpoint_dir = Path("checkpoints") / model
            config = Config.from_file(checkpoint_dir / "model_config.yaml")

        elif init == "random":
            checkpoint_dir = None
            try:
                config = Config.from_name(model)
            except ValueError:
                print(f"Model name {model} is not supported.\n")
                available_models = "\n".join(sorted(name_to_config))
                print(f"Available values:\n{available_models}")
                quit()

        else:
            raise ValueError(f"Invalid init option: {init}. Must be one of {allowed_init}")

        torch.set_float32_matmul_precision("high")
        precision = precision or get_default_supported_precision(training=False)

        fabric = L.Fabric(
            accelerator=accelerator,
            devices=devices,
            precision=precision,
        )

        if tokenizer_dir is not None:
            tokenizer_dir = extend_checkpoint_dir(Path(tokenizer_dir))
            tokenizer = Tokenizer(tokenizer_dir)
        elif checkpoint_dir is not None:
            tokenizer = Tokenizer(checkpoint_dir)
        else:
            raise ValueError("Provide a path to a tokenizer directory via the `tokenizer_dir` setting.")

        if checkpoint_dir is not None:
            prompt_style = (
                load_prompt_style(checkpoint_dir)
                if has_prompt_style(checkpoint_dir)
                else PromptStyle.from_config(config)
            )
        else:
            prompt_style = PromptStyle.from_config(config)

        with fabric.init_module(empty_init=(num_devices > 1)):
            model = GPT(config)

        with fabric.init_tensor():
            model.set_kv_cache(batch_size=1)

        model.eval()
        model = fabric.setup_module(model)

        if checkpoint_dir is not None:
            checkpoint_path = checkpoint_dir / "lit_model.pth"
            check_file_size_on_cpu_and_warn(checkpoint_path, fabric.device)
            load_checkpoint(fabric, model, checkpoint_path)
        return cls(
            model=model, tokenizer=tokenizer, devices=devices,
            prompt_style=prompt_style, checkpoint_dir=checkpoint_dir, fabric=fabric,
        )

