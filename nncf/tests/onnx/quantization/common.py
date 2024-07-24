class ModelToTest:
    def __init__(self, model_name: str, input_shape: Optional[List[int]] = None):
        self.model_name = model_name
        self.path_ref_graph = self.model_name + ".dot"
        self.input_shape = input_shape

