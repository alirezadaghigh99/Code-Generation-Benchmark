class LanguageModel:
    def __init__(
        self,
        model_name,
        hf_auth_token=None,
        device=None,
        quantization="int4",
        precision="",
        external_weights=None,
        use_system_prompt=True,
        streaming_llm=False,
    ):
        _, _, self.triple = parse_device(device)
        self.hf_model_name = llm_model_map[model_name]["hf_model_name"]
        self.device = device.split("=>")[-1].strip()
        self.backend = self.device.split("://")[0]
        self.driver = self.backend
        if "cpu" in device:
            self.device = "cpu"
            self.backend = "llvm-cpu"
            self.driver = "local-task"

        print(f"Selected {self.backend} as IREE target backend.")
        self.precision = "f32" if "cpu" in device else "f16"
        self.quantization = quantization
        self.safe_name = self.hf_model_name.replace("/", "_").replace("-", "_")
        self.external_weight_file = None
        # TODO: find a programmatic solution for model arch spec instead of hardcoding llama2
        self.file_spec = "_".join(
            [
                self.safe_name,
                self.precision,
            ]
        )
        if self.quantization != "None":
            self.file_spec += "_" + self.quantization

        if external_weights in ["safetensors", "gguf"]:
            self.external_weight_file = get_resource_path(
                os.path.join("..", self.file_spec + "." + external_weights)
            )
        else:
            self.external_weights = None
            self.external_weight_file = None

        if streaming_llm:
            # Add streaming suffix to file spec after setting external weights filename.
            self.file_spec += "_streaming"
        self.streaming_llm = streaming_llm

        self.tempfile_name = get_resource_path(
            os.path.join("..", f"{self.file_spec}.tempfile")
        )
        # TODO: Tag vmfb with target triple of device instead of HAL backend
        self.vmfb_name = str(
            get_resource_path(
                os.path.join("..", f"{self.file_spec}_{self.backend}.vmfb.tempfile")
            )
        )

        self.max_tokens = llm_model_map[model_name]["max_tokens"]
        self.iree_module_dict = None
        self.use_system_prompt = use_system_prompt
        self.global_iter = 0
        self.prev_token_len = 0
        self.first_input = True
        self.hf_auth_token = hf_auth_token
        if self.external_weight_file is not None:
            if not os.path.exists(self.external_weight_file):
                print(
                    f"External weight file {self.external_weight_file} does not exist. Generating..."
                )
                gen_external_params(
                    hf_model_name=self.hf_model_name,
                    quantization=self.quantization,
                    weight_path=self.external_weight_file,
                    hf_auth_token=hf_auth_token,
                    precision=self.precision,
                )
            else:
                print(
                    f"External weight file {self.external_weight_file} found for {self.vmfb_name}"
                )
            self.external_weight_file = str(self.external_weight_file)

        if os.path.exists(self.vmfb_name) and (
            external_weights is None or os.path.exists(str(self.external_weight_file))
        ):
            self.runner = vmfbRunner(
                device=self.driver,
                vmfb_path=self.vmfb_name,
                external_weight_path=self.external_weight_file,
            )
            if self.streaming_llm:
                self.model = self.runner.ctx.modules.streaming_state_update
            else:
                self.model = self.runner.ctx.modules.state_update
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hf_model_name,
                use_fast=False,
                use_auth_token=hf_auth_token,
            )
        elif not os.path.exists(self.tempfile_name):
            self.torch_ir, self.tokenizer = llm_model_map[self.hf_model_name][
                "initializer"
            ](
                self.hf_model_name,
                hf_auth_token,
                compile_to="torch",
                external_weights=external_weights,
                precision=self.precision,
                quantization=self.quantization,
                streaming_llm=self.streaming_llm,
                decomp_attn=True,
            )
            with open(self.tempfile_name, "w+") as f:
                f.write(self.torch_ir)
            del self.torch_ir
            gc.collect()
            self.compile()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hf_model_name,
                use_fast=False,
                use_auth_token=hf_auth_token,
            )
            self.compile()
        # Reserved for running HF torch model as reference.
        self.hf_mod = None

    def compile(self) -> None:
        # this comes with keys: "vmfb", "config", and "temp_file_to_unlink".
        # ONLY architecture/api-specific compile-time flags for each backend, if needed.
        # hf_model_id-specific global flags currently in model map.
        flags = []
        if "cpu" in self.backend:
            flags.extend(
                [
                    "--iree-global-opt-enable-quantized-matmul-reassociation",
                ]
            )
        elif self.backend == "vulkan":
            flags.extend(["--iree-stream-resource-max-allocation-size=4294967296"])
        elif self.backend == "rocm":
            flags.extend(
                [
                    "--iree-codegen-llvmgpu-enable-transform-dialect-jit=false",
                    "--iree-llvmgpu-enable-prefetch=true",
                    "--iree-opt-outer-dim-concat=true",
                    "--iree-flow-enable-aggressive-fusion",
                ]
            )
            if "gfx9" in self.triple:
                flags.extend(
                    [
                        f"--iree-codegen-transform-dialect-library={get_mfma_spec_path(self.triple, get_checkpoints_path())}",
                        "--iree-codegen-llvmgpu-use-vector-distribution=true",
                    ]
                )
        flags.extend(llm_model_map[self.hf_model_name]["compile_flags"])
        flatbuffer_blob = compile_module_to_flatbuffer(
            self.tempfile_name,
            device=self.device,
            frontend="auto",
            model_config_path=None,
            extra_args=flags,
            write_to=self.vmfb_name,
        )
        self.runner = vmfbRunner(
            device=self.driver,
            vmfb_path=self.vmfb_name,
            external_weight_path=self.external_weight_file,
        )
        if self.streaming_llm:
            self.model = self.runner.ctx.modules.streaming_state_update
        else:
            self.model = self.runner.ctx.modules.state_update

    def sanitize_prompt(self, prompt):
        if isinstance(prompt, list):
            prompt = list(chain.from_iterable(prompt))
            prompt = " ".join([x for x in prompt if isinstance(x, str)])
        prompt = prompt.replace("\n", " ")
        prompt = prompt.replace("\t", " ")
        prompt = prompt.replace("\r", " ")
        if self.use_system_prompt and self.global_iter == 0:
            prompt = append_user_prompt(DEFAULT_CHAT_SYS_PROMPT, prompt)
            return prompt
        else:
            return f"{B_INST} {prompt} {E_INST}"

    def chat(self, prompt):
        prompt = self.sanitize_prompt(prompt)

        input_tensor = self.tokenizer(prompt, return_tensors="pt").input_ids

        def format_out(results):
            return torch.tensor(results.to_host()[0][0])

        history = []
        for iter in range(self.max_tokens):
            if self.streaming_llm:
                token_slice = max(self.prev_token_len - 1, 0)
                input_tensor = input_tensor[:, token_slice:]
            if self.streaming_llm and self.model["get_seq_step"]() > 600:
                print("Evicting cache space!")
                self.model["evict_kvcache_space"]()
            token_len = input_tensor.shape[-1]
            device_inputs = [
                ireert.asdevicearray(self.runner.config.device, input_tensor)
            ]
            if self.first_input or not self.streaming_llm:
                st_time = time.time()
                token = self.model["run_initialize"](*device_inputs)
                total_time = time.time() - st_time
                token_len += 1
                self.first_input = False
            else:
                st_time = time.time()
                token = self.model["run_cached_initialize"](*device_inputs)
                total_time = time.time() - st_time
                token_len += 1

            history.append(format_out(token))
            while (
                format_out(token) != llm_model_map[self.hf_model_name]["stop_token"]
                and len(history) < self.max_tokens
            ):
                dec_time = time.time()
                if self.streaming_llm and self.model["get_seq_step"]() > 600:
                    print("Evicting cache space!")
                    self.model["evict_kvcache_space"]()
                token = self.model["run_forward"](token)
                history.append(format_out(token))
                total_time = time.time() - dec_time
                yield self.tokenizer.decode(history), total_time

            self.prev_token_len = token_len + len(history)

            if format_out(token) == llm_model_map[self.hf_model_name]["stop_token"]:
                break

        for i in range(len(history)):
            if type(history[i]) != int:
                history[i] = int(history[i])
        result_output = self.tokenizer.decode(history)
        self.global_iter += 1
        return result_output, total_time

    # Reference HF model function for sanity checks.
    def chat_hf(self, prompt):
        if self.hf_mod is None:
            self.hf_mod = AutoModelForCausalLM.from_pretrained(
                self.hf_model_name,
                torch_dtype=torch.float,
                token=self.hf_auth_token,
            )
        prompt = self.sanitize_prompt(prompt)

        input_tensor = self.tokenizer(prompt, return_tensors="pt").input_ids
        history = []
        for iter in range(self.max_tokens):
            token_len = input_tensor.shape[-1]
            if self.first_input:
                st_time = time.time()
                result = self.hf_mod(input_tensor)
                token = torch.argmax(result.logits[:, -1, :], dim=1)
                total_time = time.time() - st_time
                token_len += 1
                pkv = result.past_key_values
                self.first_input = False

            history.append(int(token))
            while token != llm_model_map[self.hf_model_name]["stop_token"]:
                dec_time = time.time()
                result = self.hf_mod(token.reshape([1, 1]), past_key_values=pkv)
                history.append(int(token))
                total_time = time.time() - dec_time
                token = torch.argmax(result.logits[:, -1, :], dim=1)
                pkv = result.past_key_values
                yield self.tokenizer.decode(history), total_time

            self.prev_token_len = token_len + len(history)

            if token == llm_model_map[self.hf_model_name]["stop_token"]:
                break
        for i in range(len(history)):
            if type(history[i]) != int:
                history[i] = int(history[i])
        result_output = self.tokenizer.decode(history)
        self.global_iter += 1
        return result_output, total_time

