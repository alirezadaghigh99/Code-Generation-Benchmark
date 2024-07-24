class FreeVCConfig(BaseVCConfig):
    """Defines parameters for FreeVC End2End TTS model.

    Args:
        model (str):
            Model name. Do not change unless you know what you are doing.

        model_args (FreeVCArgs):
            Model architecture arguments. Defaults to `FreeVCArgs()`.

        audio (FreeVCAudioConfig):
            Audio processing configuration. Defaults to `FreeVCAudioConfig()`.

        grad_clip (List):
            Gradient clipping thresholds for each optimizer. Defaults to `[1000.0, 1000.0]`.

        lr_gen (float):
            Initial learning rate for the generator. Defaults to 0.0002.

        lr_disc (float):
            Initial learning rate for the discriminator. Defaults to 0.0002.

        lr_scheduler_gen (str):
            Name of the learning rate scheduler for the generator. One of the `torch.optim.lr_scheduler.*`. Defaults to
            `ExponentialLR`.

        lr_scheduler_gen_params (dict):
            Parameters for the learning rate scheduler of the generator. Defaults to `{'gamma': 0.999875, "last_epoch":-1}`.

        lr_scheduler_disc (str):
            Name of the learning rate scheduler for the discriminator. One of the `torch.optim.lr_scheduler.*`. Defaults to
            `ExponentialLR`.

        lr_scheduler_disc_params (dict):
            Parameters for the learning rate scheduler of the discriminator. Defaults to `{'gamma': 0.999875, "last_epoch":-1}`.

        scheduler_after_epoch (bool):
            If true, step the schedulers after each epoch else after each step. Defaults to `False`.

        optimizer (str):
            Name of the optimizer to use with both the generator and the discriminator networks. One of the
            `torch.optim.*`. Defaults to `AdamW`.

        kl_loss_alpha (float):
            Loss weight for KL loss. Defaults to 1.0.

        disc_loss_alpha (float):
            Loss weight for the discriminator loss. Defaults to 1.0.

        gen_loss_alpha (float):
            Loss weight for the generator loss. Defaults to 1.0.

        feat_loss_alpha (float):
            Loss weight for the feature matching loss. Defaults to 1.0.

        mel_loss_alpha (float):
            Loss weight for the mel loss. Defaults to 45.0.

        return_wav (bool):
            If true, data loader returns the waveform as well as the other outputs. Do not change. Defaults to `True`.

        compute_linear_spec (bool):
            If true, the linear spectrogram is computed and returned alongside the mel output. Do not change. Defaults to `True`.

        use_weighted_sampler (bool):
            If true, use weighted sampler with bucketing for balancing samples between datasets used in training. Defaults to `False`.

        weighted_sampler_attrs (dict):
            Key retuned by the formatter to be used for weighted sampler. For example `{"root_path": 2.0, "speaker_name": 1.0}` sets sample probabilities
            by overweighting `root_path` by 2.0. Defaults to `{}`.

        weighted_sampler_multipliers (dict):
            Weight each unique value of a key returned by the formatter for weighted sampling.
            For example `{"root_path":{"/raid/datasets/libritts-clean-16khz-bwe-coqui_44khz/LibriTTS/train-clean-100/":1.0, "/raid/datasets/libritts-clean-16khz-bwe-coqui_44khz/LibriTTS/train-clean-360/": 0.5}`.
            It will sample instances from `train-clean-100` 2 times more than `train-clean-360`. Defaults to `{}`.

        r (int):
            Number of spectrogram frames to be generated at a time. Do not change. Defaults to `1`.

        add_blank (bool):
            If true, a blank token is added in between every character. Defaults to `True`.

        test_sentences (List[List]):
            List of sentences with speaker and language information to be used for testing.

        language_ids_file (str):
            Path to the language ids file.

        use_language_embedding (bool):
            If true, language embedding is used. Defaults to `False`.

    Note:
        Check :class:`TTS.tts.configs.shared_configs.BaseTTSConfig` for the inherited parameters.

    Example:

        >>> from TTS.vc.configs.freevc_config import FreeVCConfig
        >>> config = FreeVCConfig()
    """

    model: str = "freevc"
    # model specific params
    model_args: FreeVCArgs = field(default_factory=FreeVCArgs)
    audio: FreeVCAudioConfig = field(default_factory=FreeVCAudioConfig)

    # optimizer
    # TODO with training support

    # loss params
    # TODO with training support

    # data loader params
    return_wav: bool = True
    compute_linear_spec: bool = True

    # sampler params
    use_weighted_sampler: bool = False  # TODO: move it to the base config
    weighted_sampler_attrs: dict = field(default_factory=lambda: {})
    weighted_sampler_multipliers: dict = field(default_factory=lambda: {})

    # overrides
    r: int = 1  # DO NOT CHANGE
    add_blank: bool = True

    # multi-speaker settings
    # use speaker embedding layer
    num_speakers: int = 0
    speakers_file: str = None
    speaker_embedding_channels: int = 256

    # use d-vectors
    use_d_vector_file: bool = False
    d_vector_file: List[str] = None
    d_vector_dim: int = None

    def __post_init__(self):
        for key, val in self.model_args.items():
            if hasattr(self, key):
                self[key] = val

