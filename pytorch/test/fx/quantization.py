class MinMaxObserver:
    def __init__(self, quantizer, node):
        self.min, self.max = float("inf"), float("-inf")
        self.all_tensors = True

    def observe(self, node, env):
        v = env[node.name]
        if not isinstance(v, torch.Tensor):
            self.all_tensors = False
            return
        self.max = max(self.max, float(v.max()))
        self.min = min(self.min, float(v.min()))

    def scale_zeropoint(self):
        return _minmax_scale_zeropoint(self.min, self.max, qmin=0, qmax=255)

