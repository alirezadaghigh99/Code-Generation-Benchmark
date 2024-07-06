def compile(code):
            return torch.utils.cpp_extension.load_inline(
                name="reloaded_jit_extension",
                cpp_sources=code,
                functions="f",
                verbose=True,
            )

