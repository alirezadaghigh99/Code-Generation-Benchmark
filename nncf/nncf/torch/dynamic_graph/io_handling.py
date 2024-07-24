class FillerInputInfo(ModelInputInfo):
    """
    An implementation of ModelInputInfo that defines the model input in terms of shapes and types of individual
    tensor args and kwargs of the model's forward method.
    """

    def __init__(self, elements: List[FillerInputElement]):
        super().__init__()
        self.elements = deepcopy(elements)

    @classmethod
    def from_nncf_config(cls, config: NNCFConfig):
        """
        Parses the NNCFConfig's "input_info" field if it is present to determine model input information,
        otherwise raises a RuntimeError. The "input_info" field structure must conform to the NNCF config jsonschema.

        :param config: An NNCFConfig instance.
        :return: FillerInputInfo object initialized according to config.
        """
        input_infos = config.get("input_info")
        if input_infos is None:
            raise nncf.ValidationError("Passed NNCFConfig does not have an 'input_info' field")
        if isinstance(input_infos, dict):
            return FillerInputInfo(
                [
                    FillerInputElement(
                        input_infos.get("sample_size"),
                        input_infos.get("type"),
                        input_infos.get("keyword"),
                        input_infos.get("filler"),
                    )
                ]
            )
        if isinstance(input_infos, list):
            elements: List[FillerInputElement] = []
            for info_dict in input_infos:
                elements.append(
                    FillerInputElement(
                        info_dict.get("sample_size"),
                        info_dict.get("type"),
                        info_dict.get("keyword"),
                        info_dict.get("filler"),
                    )
                )
            return FillerInputInfo(elements)
        raise nncf.ValidationError("Invalid input_infos specified in config - should be either dict or list of dicts")

    def get_forward_inputs(
        self, device: Optional[Union[str, torch.device]] = None
    ) -> Tuple[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        args_list = []
        kwargs = {}
        for fe in self.elements:
            tensor = fe.get_tensor_for_input()
            if device is not None:
                tensor = tensor.to(device)
            if fe.keyword is None:
                args_list.append(tensor)
            else:
                kwargs[fe.keyword] = tensor
        return tuple(args_list), kwargs

class FillerInputElement:
    """
    Represents a single tensor argument (positional or keyword) in the model's example input that is to be generated
    on the fly and filled with a requested type of data filler.
    """

    FILLER_TYPE_ONES = "ones"
    FILLER_TYPE_ZEROS = "zeros"
    FILLER_TYPE_RANDOM = "random"
    FILLER_TYPES = [FILLER_TYPE_ONES, FILLER_TYPE_ZEROS, FILLER_TYPE_RANDOM]

    def __init__(self, shape: List[int], type_str: str = "float", keyword: str = None, filler: str = None):
        """
        :param shape: The shape of the model input tensor.
        :param type_str: The type of the model input tensor - "float" for torch.float32, "long" for torch.long
        :param keyword: Optional - if specified, then this input tensor will be passed as a corresponding keyword
          parameter, and as a positional argument if this parameter is unspecified.
        :param filler: Optional - can be either "ones", "zeros" or "random". The model input tensor will be generated
          with data corresponding to this setting. Default is "ones".
        """
        self.shape = shape
        self.type = self._string_to_torch_type(type_str)
        self.keyword = keyword
        if filler is None:
            self.filler = self.FILLER_TYPE_ONES
        else:
            self.filler = filler
            if self.filler not in self.FILLER_TYPES:
                raise ValueError(f"Unknown input filler type: {filler}")

    @staticmethod
    def _string_to_torch_type(string):
        if string == "long":
            return torch.long
        return torch.float32

    @staticmethod
    def torch_type_to_string(dtype: torch.dtype):
        if dtype is torch.long:
            return "long"
        return "float"

    def is_integer_input(self):
        return self.type != torch.float32

    def __eq__(self, other: "FillerInputElement"):
        return self.type == other.type and self.keyword == other.keyword and self.filler == other.filler

    def get_tensor_for_input(self) -> torch.Tensor:
        if self.filler == FillerInputElement.FILLER_TYPE_ZEROS:
            return torch.zeros(size=self.shape, dtype=self.type)
        if self.filler == FillerInputElement.FILLER_TYPE_ONES:
            return torch.ones(size=self.shape, dtype=self.type)
        if self.filler == FillerInputElement.FILLER_TYPE_RANDOM:
            return torch.rand(size=self.shape, dtype=self.type)
        raise NotImplementedError

