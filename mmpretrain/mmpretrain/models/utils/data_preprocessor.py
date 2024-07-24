class VideoDataPreprocessor(BaseDataPreprocessor):
    """Video pre-processor for operations, like normalization and bgr to rgb
    conversion .

    Compared with the :class:`mmaction.ActionDataPreprocessor`, this module
    supports ``inputs`` as torch.Tensor or a list of torch.Tensor.

    Args:
        mean (Sequence[float or int, optional): The pixel mean of channels
            of images or stacked optical flow. Defaults to None.
        std (Sequence[float or int], optional): The pixel standard deviation
            of channels of images or stacked optical flow. Defaults to None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (float or int): The padded pixel value. Defaults to 0.
        to_rgb (bool): Whether to convert image from BGR to RGB.
            Defaults to False.
        format_shape (str): Format shape of input data.
            Defaults to ``'NCHW'``.
    """

    def __init__(self,
                 mean: Optional[Sequence[Union[float, int]]] = None,
                 std: Optional[Sequence[Union[float, int]]] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 to_rgb: bool = False,
                 format_shape: str = 'NCHW') -> None:
        super().__init__()
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value
        self.to_rgb = to_rgb
        self.format_shape = format_shape

        if mean is not None:
            assert std is not None, 'To enable the normalization in ' \
                                    'preprocessing, please specify both ' \
                                    '`mean` and `std`.'
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            if self.format_shape == 'NCHW':
                normalizer_shape = (-1, 1, 1)
            elif self.format_shape == 'NCTHW':
                normalizer_shape = (-1, 1, 1, 1)
            else:
                raise ValueError(f'Invalid format shape: {format_shape}')

            self.register_buffer(
                'mean',
                torch.tensor(mean, dtype=torch.float32).view(normalizer_shape),
                False)
            self.register_buffer(
                'std',
                torch.tensor(std, dtype=torch.float32).view(normalizer_shape),
                False)
        else:
            self._enable_normalize = False

    def forward(
            self,
            data: dict,
            training: bool = False
    ) -> Tuple[List[torch.Tensor], Optional[list]]:
        """Performs normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation. If
                subclasses override this method, they can perform different
                preprocessing strategies for training and testing based on the
                value of ``training``.
        Returns:
            Tuple[List[torch.Tensor], Optional[list]]: Data in the same format
                as the model input.
        """

        data = [val for _, val in data.items()]
        batch_inputs, batch_data_samples = self.cast_data(data)

        if isinstance(batch_inputs, list):
            # channel transform
            if self.to_rgb:
                if self.format_shape == 'NCHW':
                    batch_inputs = [
                        _input[..., [2, 1, 0], :, :] for _input in batch_inputs
                    ]
                elif self.format_shape == 'NCTHW':
                    batch_inputs = [
                        _input[..., [2, 1, 0], :, :, :]
                        for _input in batch_inputs
                    ]
                else:
                    raise ValueError(
                        f'Invalid format shape: {self.format_shape}')

            # convert to float after channel conversion to ensure efficiency
            batch_inputs = [_input.float() for _input in batch_inputs]

            # normalization
            if self._enable_normalize:
                batch_inputs = [(_input - self.mean) / self.std
                                for _input in batch_inputs]

        else:
            # channel transform
            if self.to_rgb:
                if self.format_shape == 'NCHW':
                    batch_inputs = batch_inputs[..., [2, 1, 0], :, :]
                elif self.format_shape == 'NCTHW':
                    batch_inputs = batch_inputs[..., [2, 1, 0], :, :, :]
                else:
                    raise ValueError(
                        f'Invalid format shape: {self.format_shape}')

            # convert to float after channel conversion to ensure efficiency
            batch_inputs = batch_inputs.float()

            # normalization
            if self._enable_normalize:
                batch_inputs = (batch_inputs - self.mean) / self.std

        return {'inputs': batch_inputs, 'data_samples': batch_data_samples}

