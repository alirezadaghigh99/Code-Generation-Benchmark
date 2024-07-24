class QuantizeTestCaseConfiguration:
    def __init__(self, quant_params, graph_dir):
        a_mode, a_per_channel = quant_params["activations"]
        w_mode, w_per_channel = quant_params["weights"]
        self.a_mode = a_mode
        self.w_mode = w_mode
        self.a_per_channel = a_per_channel == "per_channel"
        self.w_per_channel = w_per_channel == "per_channel"
        self.graph_dir = graph_dir

