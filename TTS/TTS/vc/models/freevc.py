class FreeVC(BaseVC):
    """

    Papaer::
        https://arxiv.org/abs/2210.15418#

    Paper Abstract::
        Voice conversion (VC) can be achieved by first extracting source content information and target speaker
        information, and then reconstructing waveform with these information. However, current approaches normally
        either extract dirty content information with speaker information leaked in, or demand a large amount of
        annotated data for training. Besides, the quality of reconstructed waveform can be degraded by the
        mismatch between conversion model and vocoder. In this paper, we adopt the end-to-end framework of VITS for
        high-quality waveform reconstruction, and propose strategies for clean content information extraction without
        text annotation. We disentangle content information by imposing an information bottleneck to WavLM features,
        and propose the spectrogram-resize based data augmentation to improve the purity of extracted content
        information. Experimental results show that the proposed method outperforms the latest VC models trained with
        annotated data and has greater robustness.

    Original Code::
        https://github.com/OlaWod/FreeVC

    Examples:
        >>> from TTS.vc.configs.freevc_config import FreeVCConfig
        >>> from TTS.vc.models.freevc import FreeVC
        >>> config = FreeVCConfig()
        >>> model = FreeVC(config)
    """

    def __init__(self, config: Coqpit, speaker_manager: SpeakerManager = None):
        super().__init__(config, None, speaker_manager, None)

        self.init_multispeaker(config)

        self.spec_channels = self.args.spec_channels
        self.inter_channels = self.args.inter_channels
        self.hidden_channels = self.args.hidden_channels
        self.filter_channels = self.args.filter_channels
        self.n_heads = self.args.n_heads
        self.n_layers = self.args.n_layers
        self.kernel_size = self.args.kernel_size
        self.p_dropout = self.args.p_dropout
        self.resblock = self.args.resblock
        self.resblock_kernel_sizes = self.args.resblock_kernel_sizes
        self.resblock_dilation_sizes = self.args.resblock_dilation_sizes
        self.upsample_rates = self.args.upsample_rates
        self.upsample_initial_channel = self.args.upsample_initial_channel
        self.upsample_kernel_sizes = self.args.upsample_kernel_sizes
        self.segment_size = self.args.segment_size
        self.gin_channels = self.args.gin_channels
        self.ssl_dim = self.args.ssl_dim
        self.use_spk = self.args.use_spk

        self.enc_p = Encoder(self.args.ssl_dim, self.inter_channels, self.hidden_channels, 5, 1, 16)
        self.dec = Generator(
            self.inter_channels,
            self.resblock,
            self.resblock_kernel_sizes,
            self.resblock_dilation_sizes,
            self.upsample_rates,
            self.upsample_initial_channel,
            self.upsample_kernel_sizes,
            gin_channels=self.gin_channels,
        )
        self.enc_q = Encoder(
            self.spec_channels, self.inter_channels, self.hidden_channels, 5, 1, 16, gin_channels=self.gin_channels
        )
        self.flow = ResidualCouplingBlock(
            self.inter_channels, self.hidden_channels, 5, 1, 4, gin_channels=self.gin_channels
        )
        if not self.use_spk:
            self.enc_spk = SpeakerEncoder(model_hidden_size=self.gin_channels, model_embedding_size=self.gin_channels)
        else:
            self.load_pretrained_speaker_encoder()

        self.wavlm = get_wavlm()

    @property
    def device(self):
        return next(self.parameters()).device

    def load_pretrained_speaker_encoder(self):
        """Load pretrained speaker encoder model as mentioned in the paper."""
        print(" > Loading pretrained speaker encoder model ...")
        self.enc_spk_ex = SpeakerEncoderEx(
            "https://github.com/coqui-ai/TTS/releases/download/v0.13.0_models/speaker_encoder.pt"
        )

    def init_multispeaker(self, config: Coqpit):
        """Initialize multi-speaker modules of a model. A model can be trained either with a speaker embedding layer
        or with external `d_vectors` computed from a speaker encoder model.

        You must provide a `speaker_manager` at initialization to set up the multi-speaker modules.

        Args:
            config (Coqpit): Model configuration.
            data (List, optional): Dataset items to infer number of speakers. Defaults to None.
        """
        self.num_spks = self.args.num_spks
        if self.speaker_manager:
            self.num_spks = self.speaker_manager.num_spks

    def forward(
        self,
        c: torch.Tensor,
        spec: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        mel: Optional[torch.Tensor] = None,
        c_lengths: Optional[torch.Tensor] = None,
        spec_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Forward pass of the model.

        Args:
            c: WavLM features. Shape: (batch_size, c_seq_len).
            spec: The input spectrogram. Shape: (batch_size, spec_seq_len, spec_dim).
            g: The speaker embedding. Shape: (batch_size, spk_emb_dim).
            mel: The input mel-spectrogram for the speaker encoder. Shape: (batch_size, mel_seq_len, mel_dim).
            c_lengths: The lengths of the WavLM features. Shape: (batch_size,).
            spec_lengths: The lengths of the spectrogram. Shape: (batch_size,).

        Returns:
            o: The output spectrogram. Shape: (batch_size, spec_seq_len, spec_dim).
            ids_slice: The slice indices. Shape: (batch_size, num_slices).
            spec_mask: The spectrogram mask. Shape: (batch_size, spec_seq_len).
            (z, z_p, m_p, logs_p, m_q, logs_q): A tuple of latent variables.
        """

        # If c_lengths is None, set it to the length of the last dimension of c
        if c_lengths is None:
            c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)

        # If spec_lengths is None, set it to the length of the last dimension of spec
        if spec_lengths is None:
            spec_lengths = (torch.ones(spec.size(0)) * spec.size(-1)).to(spec.device)

        # If use_spk is False, compute g from mel using enc_spk
        g = None
        if not self.use_spk:
            g = self.enc_spk(mel).unsqueeze(-1)

        # Compute m_p, logs_p, z, m_q, logs_q, and spec_mask using enc_p and enc_q
        _, m_p, logs_p, _ = self.enc_p(c, c_lengths)
        z, m_q, logs_q, spec_mask = self.enc_q(spec.transpose(1, 2), spec_lengths, g=g)

        # Compute z_p using flow
        z_p = self.flow(z, spec_mask, g=g)

        # Randomly slice z and compute o using dec
        z_slice, ids_slice = commons.rand_slice_segments(z, spec_lengths, self.segment_size)
        o = self.dec(z_slice, g=g)

        return o, ids_slice, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    @torch.no_grad()
    def inference(self, c, g=None, mel=None, c_lengths=None):
        """
        Inference pass of the model

        Args:
            c (torch.Tensor): Input tensor. Shape: (batch_size, c_seq_len).
            g (torch.Tensor): Speaker embedding tensor. Shape: (batch_size, spk_emb_dim).
            mel (torch.Tensor): Mel-spectrogram tensor. Shape: (batch_size, mel_seq_len, mel_dim).
            c_lengths (torch.Tensor): Lengths of the input tensor. Shape: (batch_size,).

        Returns:
            torch.Tensor: Output tensor.
        """
        if c_lengths == None:
            c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)
        if not self.use_spk:
            g = self.enc_spk.embed_utterance(mel)
            g = g.unsqueeze(-1)
        z_p, m_p, logs_p, c_mask = self.enc_p(c, c_lengths)
        z = self.flow(z_p, c_mask, g=g, reverse=True)
        o = self.dec(z * c_mask, g=g)
        return o

    def extract_wavlm_features(self, y):
        """Extract WavLM features from an audio tensor.

        Args:
            y (torch.Tensor): Audio tensor. Shape: (batch_size, audio_seq_len).
        """

        with torch.no_grad():
            c = self.wavlm.extract_features(y)[0]
        c = c.transpose(1, 2)
        return c

    def load_audio(self, wav):
        """Read and format the input audio."""
        if isinstance(wav, str):
            wav, _ = librosa.load(wav, sr=self.config.audio.input_sample_rate)
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav).to(self.device)
        if isinstance(wav, torch.Tensor):
            wav = wav.to(self.device)
        if isinstance(wav, list):
            wav = torch.from_numpy(np.array(wav)).to(self.device)
        return wav.float()

    @torch.inference_mode()
    def voice_conversion(self, src, tgt):
        """
        Voice conversion pass of the model.

        Args:
            src (str or torch.Tensor): Source utterance.
            tgt (str or torch.Tensor): Target utterance.

        Returns:
            torch.Tensor: Output tensor.
        """

        wav_tgt = self.load_audio(tgt).cpu().numpy()
        wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)

        if self.config.model_args.use_spk:
            g_tgt = self.enc_spk_ex.embed_utterance(wav_tgt)
            g_tgt = torch.from_numpy(g_tgt)[None, :, None].to(self.device)
        else:
            wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).to(self.device)
            mel_tgt = mel_spectrogram_torch(
                wav_tgt,
                self.config.audio.filter_length,
                self.config.audio.n_mel_channels,
                self.config.audio.input_sample_rate,
                self.config.audio.hop_length,
                self.config.audio.win_length,
                self.config.audio.mel_fmin,
                self.config.audio.mel_fmax,
            )
        # src
        wav_src = self.load_audio(src)
        c = self.extract_wavlm_features(wav_src[None, :])

        if self.config.model_args.use_spk:
            audio = self.inference(c, g=g_tgt)
        else:
            audio = self.inference(c, mel=mel_tgt.transpose(1, 2))
        audio = audio[0][0].data.cpu().float().numpy()
        return audio

    def eval_step():
        ...

    @staticmethod
    def init_from_config(config: FreeVCConfig, samples: Union[List[List], List[Dict]] = None, verbose=True):
        model = FreeVC(config)
        return model

    def load_checkpoint(self, config, checkpoint_path, eval=False, strict=True, cache=False):
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"), cache=cache)
        self.load_state_dict(state["model"], strict=strict)
        if eval:
            self.eval()

    def train_step():
        ...

