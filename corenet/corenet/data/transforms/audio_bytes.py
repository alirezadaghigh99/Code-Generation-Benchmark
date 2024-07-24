class TorchaudioSave(BaseTransformation):
    """
    Encode audio with a supported file encoding.

    Args:
        opts: The global options.
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts=opts, *args, **kwargs)
        self.opts = opts
        self.encoding_dtype = getattr(
            self.opts, "audio_augmentation.torchaudio_save.encoding_dtype"
        )
        self.format = getattr(self.opts, "audio_augmentation.torchaudio_save.format")
        self.backend = getattr(self.opts, "audio_augmentation.torchaudio_save.backend")

    def __call__(
        self, data: Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor, int]]
    ) -> Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor, int]]:
        """
        Serialize the input as file bytes.

        Args:
            data: A tensor of the form:
                {
                    "samples": {"audio": tensor of shape [num_channels, sequence_length]},
                    "metadata": {"audio_fps": the audio framerate.}
                }

        Returns:
            The transformed data.
        """
        x = data["samples"]["audio"]
        audio_fps = data["metadata"]["audio_fps"]
        if x.dim() == 2:
            # @x is [C, N] in shape. Convert to mono.
            if x.shape[0] in (1, 2):
                x = x.mean(dim=0)
            else:
                raise ValueError(f"Expected x.shape[0] to be 1 or 2, got {x.shape}")
        else:
            raise ValueError(f"Expected x.dim() == 2, got shape {x.shape}")

        if self.format == "wav":
            file_bytes = _stream_to_wav(x, self.encoding_dtype, audio_fps, self.backend)
            buf = np.frombuffer(file_bytes, dtype=np.uint8)
            # Convert to int32 so we can use negative values as padding.
            # The copy operation is required to avoid a warning about non-writable
            # tensors.
            buf = torch.from_numpy(buf.copy()).to(dtype=torch.int32)
            data["samples"]["audio"] = buf

        elif self.format == "mp3":
            if x.dim() == 1:
                x = x.reshape(1, -1)
            with tempfile.NamedTemporaryFile("rb+", suffix=".mp3") as f:
                # The sox backend does not support writing to BytesIO.
                torchaudio.save(f.name, x, audio_fps, backend=self.backend)
                byte_values = f.read()
            buf = np.frombuffer(byte_values, dtype=np.uint8)
            # Convert to int32 so we can use negative values as padding.
            # The copy operation is required to avoid a warning about non-writable
            # tensors.
            buf = torch.from_numpy(buf.copy()).to(dtype=torch.int32)
            data["samples"]["audio"] = buf

        else:
            raise NotImplementedError(
                f"Format {self.format} not implemented. Only 'wav' and 'mp3' are supported."
            )

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--audio-augmentation.torchaudio-save.enable",
            action="store_true",
            help="Use {}. This flag is useful when you want to study the effect of different "
            "transforms.".format(cls.__name__),
        )
        group.add_argument(
            "--audio-augmentation.torchaudio-save.encoding-dtype",
            choices=("float32", "int32", "int16", "uint8"),
            help="The data type used in the audio encoding. Defaults to float32.",
            default="float32",
        )
        group.add_argument(
            "--audio-augmentation.torchaudio-save.format",
            choices=("wav", "mp3"),
            default="wav",
            help="The format in which to save the audio. Defaults to wav.",
        )
        group.add_argument(
            "--audio-augmentation.torchaudio-save.backend",
            choices=("ffmpeg", "sox", "soundfile"),
            default="sox",
            help=(
                "The I/O backend to use for save the audio. Defaults to sox, which was"
                " the default backend in the earlier torchaudio versions."
            ),
        )
        return parser

