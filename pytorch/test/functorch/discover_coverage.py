def all(cls):
        dct = get_public_overridable_outplace_we_care_about()
        names = dct.keys()
        names_sanitized = []
        for n in names:
            torch_tensor = "torch.Tensor."
            torch_dot = "torch."
            if n.startswith(torch_tensor):
                names_sanitized.append(n[len(torch_tensor) :])
            elif n.startswith(torch_dot):
                names_sanitized.append(n[len(torch_dot) :])
            else:
                raise AssertionError
        return cls.from_names(names_sanitized)

