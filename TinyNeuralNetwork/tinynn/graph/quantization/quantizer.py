    def convert(self, q_model: nn.Module, backend: str = 'tflite') -> nn.Module:
        """Converts a QAT/PTQ-prepared model to an actual quantized model

        Args:
            q_model (nn.Module): The QAT/PTQ-prepared model
            backend (str): The backend to translate for, including `pytorch` and `tflite`. Defaults to `tflite`

        Returns:
            nn.Module: The QAT/PTQ-converted model. When the backend is set to `pytorch`, it is used for validation \
                in PyTorch only.
        """

        for acp, post_acp, dq_name, q_name, activ_name, activ_type in self.extra_qparams_mappings:
            if backend != 'pytorch' and activ_type in ('relu', 'relu6', torch.nn.ReLU, torch.nn.ReLU6):
                continue

            acp.scale = post_acp.scale
            acp.zero_point = post_acp.zero_point
            acp.activation_post_process.min_val = post_acp.activation_post_process.min_val
            acp.activation_post_process.max_val = post_acp.activation_post_process.max_val

            setattr(q_model, dq_name, nn.Identity())
            setattr(q_model, q_name, nn.Identity())
            if activ_name is not None:
                setattr(q_model, activ_name, nn.Identity())

        if type(self).__name__ == 'QATQuantizer':

            q = queue.Queue()
            q.put(q_model)

            while not q.empty():
                m = q.get()
                for n, c in m.named_children():
                    if isinstance(c, ConvTransposeBn2d):
                        setattr(m, n, c.transform(c))
                    else:
                        q.put(c)

            if hasattr(torch_q, 'get_default_static_quant_module_mappings'):
                mapping = torch_q.get_default_static_quant_module_mappings()
            elif hasattr(torch_q, 'get_static_quant_module_mappings'):
                mapping = copy.deepcopy(torch_q.get_static_quant_module_mappings())
            else:
                mapping = copy.deepcopy(torch_q.DEFAULT_MODULE_MAPPING)

            mapping.update(FUSE_QAT_MODULES_CVT)

            float_mods = {}

            for qat_t, q_t in FUSE_QAT_MODULES_CVT.items():
                float_mod = getattr(q_t, '_FLOAT_MODULE', None)
                if float_mod is not None:
                    float_mods[q_t] = float_mod
                    setattr(q_t, '_FLOAT_MODULE', qat_t)

            q_model = torch.quantization.convert(q_model, mapping)

            for q_t, orig_t in float_mods.items():
                setattr(q_t, '_FLOAT_MODULE', orig_t)

            float_mods.clear()
        else:
            q_model = torch.quantization.convert(q_model)

        return q_model