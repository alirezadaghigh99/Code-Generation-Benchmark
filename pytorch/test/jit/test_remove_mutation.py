def normal():
            # NOTE: For some unknown reason, the
            # `torch._C._jit_pass_remove_mutation` call within `self.run_pass`
            # replaces `torch.randn(..., dtype=None).normal_()` with an
            # `aten::normal` call with dtype double, even if the default dtype
            # is float. So we must explicitly set the dtype here
            return torch.rand(2, 1, 3, 4, dtype=torch.float).normal_()

