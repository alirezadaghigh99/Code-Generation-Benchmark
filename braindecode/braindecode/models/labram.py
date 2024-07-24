class Labram(EEGModuleMixin, nn.Module):
    """Labram.

    Large Brain Model for Learning Generic Representations with Tremendous
    EEG Data in BCI from [Jiang2024]_

    This an **adaptation** of the code [Code2024]_ from the Labram model.

    The model is transformer architecture with **strong** inspiration from
    BEiTv2 [BeiTv2]_.

    The models can be used in two modes:
        - Neural Tokenizor: Design to get an embedding layers (e.g. classification).
        - Neural Decoder: To extract the ampliture and phase outputs with a VQSNP.

    The braindecode's modification is to allow the model to be used in
    with an input shape of (batch, n_chans, n_times), if neural tokenizer
    equals True. The original implementation uses (batch, n_chans, n_patches,
    patch_size) as input with static segmentation of the input data.

    The models have the following sequence of steps:
    if neural tokenizer:
        - SegmentPatch: Segment the input data in patches;
        - TemporalConv: Apply a temporal convolution to the segmented data;
        - Residual adding cls, temporal and position embeddings (optional);
        - WindowsAttentionBlock: Apply a windows attention block to the data;
        - LayerNorm: Apply layer normalization to the data;
        - Linear: An head linear layer to transformer the data into classes.
    else:
        - PatchEmbed: Apply a patch embedding to the input data;
        - Residual adding cls, temporal and position embeddings (optional);
        - WindowsAttentionBlock: Apply a windows attention block to the data;
        - LayerNorm: Apply layer normalization to the data;
        - Linear: An head linear layer to transformer the data into classes.

    .. versionadded:: 0.9

    Parameters
    ----------
    patch_size : int
        The size of the patch to be used in the patch embedding.
    emb_size : int
        The dimension of the embedding.
    in_channels : int
        The number of convolutional input channels.
    out_channels : int
        The number of convolutional output channels.
    n_layers :  int (default=12)
        The number of attention layers of the model.
    att_num_heads : int (default=10)
        The number of attention heads.
    mlp_ratio : float (default=4.0)
        The expansion ratio of the mlp layer
    qkv_bias :  bool (default=False)
        If True, add a learnable bias to the query, key, and value tensors.
    qk_norm : Pytorch Normalize layer (default=None)
        If not None, apply LayerNorm to the query and key tensors
    qk_scale : float (default=None)
        If not None, use this value as the scale factor. If None,
        use head_dim**-0.5, where head_dim = dim // num_heads.
    drop_rate : float (default=0.0)
        Dropout rate for the attention weights.
    attn_drop_rate : float (default=0.0)
        Dropout rate for the attention weights.
    drop_path_rate : float (default=0.0)
        Dropout rate for the attention weights used on DropPath.
    norm_layer : Pytorch Normalize layer (default=nn.LayerNorm)
        The normalization layer to be used.
    init_values : float (default=None)
        If not None, use this value to initialize the gamma_1 and gamma_2
        parameters.
    use_abs_pos_emb : bool (default=True)
        If True, use absolute position embedding.
    use_mean_pooling : bool (default=True)
        If True, use mean pooling.
    init_scale : float (default=0.001)
        The initial scale to be used in the parameters of the model.
    neural_tokenizer : bool (default=True)
        The model can be used in two modes: Neural Tokenizor or Neural Decoder.
    attn_head_dim : bool (default=None)
        The head dimension to be used in the attention layer, to be used only
        during pre-training.
    References
    ----------
    .. [Jiang2024] Wei-Bang Jiang, Li-Ming Zhao, Bao-Liang Lu. 2024, May.
       Large Brain Model for Learning Generic Representations with Tremendous
       EEG Data in BCI. The Twelfth International Conference on Learning
       Representations, ICLR.
    .. [Code2024] Wei-Bang Jiang, Li-Ming Zhao, Bao-Liang Lu. 2024. Labram
       Large Brain Model for Learning Generic Representations with Tremendous
       EEG Data in BCI. GitHub https://github.com/935963004/LaBraM
       (accessed 2024-03-02)
    .. [BeiTv2] Zhiliang Peng, Li Dong, Hangbo Bao, Qixiang Ye, Furu Wei. 2024.
       BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers.
       arXiv:2208.06366 [cs.CV]
    """

    def __init__(
        self,
        n_times=None,
        n_outputs=None,
        chs_info=None,
        n_chans=None,
        sfreq=None,
        input_window_seconds=None,
        patch_size=200,
        emb_size=200,
        in_channels=1,
        out_channels=8,
        n_layers=12,
        att_num_heads=10,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=None,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=None,
        use_abs_pos_emb=True,
        use_mean_pooling=True,
        init_scale=0.001,
        neural_tokenizer=True,
        attn_head_dim=None,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        self.patch_size = patch_size
        self.n_path = self.n_times // patch_size
        self.num_features = self.emb_size = emb_size
        self.neural_tokenizer = neural_tokenizer
        self.init_scale = init_scale

        if neural_tokenizer and in_channels != 1:
            warn(
                "The model is in Neural Tokenizer mode, but the variable "
                + "`in_channels` is different from the default values."
                + "`in_channels` is only needed for the Neural Decoder mode."
                + "in_channels is not used in the Neural Tokenizer mode.",
                UserWarning,
            )
            in_channels = 1
            # If you can use the model in Neural Tokenizer mode,
        # temporal conv layer will be use over the patched dataset
        if neural_tokenizer:
            self.patch_embed = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "segment_patch",
                            _SegmentPatch(
                                n_times=self.n_times,
                                patch_size=self.patch_size,
                                n_chans=self.n_chans,
                                emb_dim=self.patch_size,
                            ),
                        ),
                        ("temporal_conv", _TemporalConv(out_channels=out_channels)),
                    ]
                )
            )
        else:
            # If not, the model will be used as Neural Decoder mode
            # So the input here will be after the VQVAE encoder
            # To be used to extract the ampliture and phase outputs.
            # Adding inside a Sequential to use the same convention as the
            # Neural Tokenizer mode.
            self.patch_embed = nn.Sequential()
            self.patch_embed.add_module(
                "segment_patch",
                _PatchEmbed(
                    n_times=self.n_times,
                    patch_size=patch_size,
                    in_channels=in_channels,
                    emb_dim=self.emb_size,
                ),
            )
        # Defining the parameters
        # Creating a parameter list with cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.emb_size))
        # Positional embedding and time embedding are complementary
        # one is for the spatial information and the other is for the temporal
        # information.
        # The time embedding is used to encode something in the number of
        # patches, and the position embedding is used to encode the channels'
        # information.
        if use_abs_pos_emb:
            self.position_embedding = nn.Parameter(
                torch.zeros(1, self.n_chans + 1, self.emb_size),
                requires_grad=True,
            )
        else:
            self.position_embedding = None

        self.temporal_embedding = nn.Parameter(
            torch.zeros(1, self.patch_embed[0].n_patchs + 1, self.emb_size),
            requires_grad=True,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, n_layers)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                _WindowsAttentionBlock(
                    dim=self.emb_size,
                    num_heads=att_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                    window_size=(
                        self.patch_embed[0].patch_shape
                        if not neural_tokenizer
                        else None
                    ),
                    attn_head_dim=attn_head_dim,
                )
                for i in range(n_layers)
            ]
        )
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(self.emb_size)
        self.fc_norm = norm_layer(self.emb_size) if use_mean_pooling else None

        if self.n_outputs > 0:
            self.head = nn.Linear(self.emb_size, self.n_outputs)
        else:
            self.head = nn.Identity()

        self.apply(self._init_weights)
        self.fix_init_weight_and_init_embedding()

    def fix_init_weight_and_init_embedding(self):
        """
        Fix the initial weight and the initial embedding.
        Initializing with truncated normal distribution.
        """
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.temporal_embedding, std=0.02)

        if self.position_embedding is not None:
            trunc_normal_(self.position_embedding, std=0.02)

        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=0.02)

        for layer_id, layer in enumerate(self.blocks):
            rescale_parameter(layer.attn.proj.weight.data, layer_id + 1)
            rescale_parameter(layer.mlp[-2].weight.data, layer_id + 1)

        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(self.init_scale)
            self.head.bias.data.mul_(self.init_scale)

    @staticmethod
    def _init_weights(layer):
        """
        Initialize the weights of the model for each layer layer.

        If the layer is a linear layer, the weight will be initialized
        with a truncated normal distribution with std=0.02.

        If m.bias is not None, the bias will be initialized with a constant
        value of 0.

        If the layer is a layer normalization layer, the bias will be
        initialized with a constant value of 0, and the weight will be
        initialized with a constant value of 1.

        Parameters
        ----------
        m : torch.nn.Module
            The layer of the pytorch model
        """

        if isinstance(layer, nn.Linear):
            trunc_normal_(layer.weight, std=0.02)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.LayerNorm):
            nn.init.constant_(layer.bias, 0)
            nn.init.constant_(layer.weight, 1.0)

    def get_num_layers(self):
        """
        Convenience method to get the number of layers in the model.
        """
        return len(self.blocks)

    def forward_features(
        self,
        x,
        input_chans=None,
        return_patch_tokens=False,
        return_all_tokens=False,
    ):
        """
        Forward the features of the model.

        Parameters
        ----------
        x : torch.Tensor
            The input data with shape (batch, n_chans, n_patches, patch size),
            if neural decoder or (batch, n_chans, n_times), if neural tokenizer.
        input_chans : int
            The number of input channels.
        return_patch_tokens : bool
            Whether to return the patch tokens.
        return_all_tokens : bool
            Whether to return all the tokens.

        Returns
        -------
        x : torch.Tensor
            The output of the model.
        """

        if self.neural_tokenizer:
            batch_size, nch, n_patch, temporal = self.patch_embed.segment_patch(x).shape
        else:
            batch_size, nch, n_patch = self.patch_embed(x).shape
        x = self.patch_embed(x)
        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)

        # Positional Embedding
        if input_chans is not None:
            pos_embed_used = self.position_embedding[:, input_chans]
        else:
            pos_embed_used = self.position_embedding

        if self.position_embedding is not None:
            pos_embed = self._adj_position_embedding(
                pos_embed_used=pos_embed_used, batch_size=batch_size
            )
            x += pos_embed

        # The time embedding is added across the channels after the [CLS] token
        if self.neural_tokenizer:
            num_ch = self.n_chans
        else:
            num_ch = n_patch
        time_embed = self._adj_temporal_embedding(
            num_ch=num_ch, batch_size=batch_size, dim_embed=temporal
        )
        x[:, 1:, :] += time_embed

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        if self.fc_norm is not None:
            if return_all_tokens:
                return self.fc_norm(x)
            temporal = x[:, 1:, :]
            if return_patch_tokens:
                return self.fc_norm(temporal)
            return self.fc_norm(temporal.mean(1))
        else:
            if return_all_tokens:
                return x
            elif return_patch_tokens:
                return x[:, 1:]
            return x[:, 0]

    def forward(
        self,
        x,
        input_chans=None,
        return_patch_tokens=False,
        return_all_tokens=False,
    ):
        """
        Forward the input EEG data through the model.

        Parameters
        ----------
        x: torch.Tensor
            The input data with shape (batch, n_chans, n_times)
            or (batch, n_chans, n_patches, patch size).
        input_chans: int
            An input channel to select some dimensions
        return_patch_tokens: bool
            Return the patch tokens
        return_all_tokens: bool
            Return all the tokens

        Returns
        -------
        torch.Tensor
            The output of the model with dimensions (batch, n_outputs)
        """
        x = self.forward_features(
            x,
            input_chans=input_chans,
            return_patch_tokens=return_patch_tokens,
            return_all_tokens=return_all_tokens,
        )
        x = self.head(x)
        return x

    def get_classifier(self):
        """
        Get the classifier of the model.

        Returns
        -------
        torch.nn.Module
            The classifier of the head model.
        """
        return self.head

    def reset_classifier(self, n_outputs):
        """
        Reset the classifier with the new number of classes.

        Parameters
        ----------
        n_outputs : int
            The new number of classes.
        """
        self.n_outputs = n_outputs
        self.head = (
            nn.Linear(self.emb_dim, self.n_outputs)
            if self.n_outputs > 0
            else nn.Identity()
        )

    def _adj_temporal_embedding(self, num_ch, batch_size, dim_embed=None):
        """
        Adjust the dimensions of the time embedding to match the
        number of channels.

        Parameters
        ----------
        num_ch : int
            The number of channels or number of code books vectors.
        batch_size : int
            Batch size of the input data.

        Returns
        -------
        temporal_embedding : torch.Tensor
            The adjusted time embedding to be added across the channels
            after the [CLS] token. (x[:, 1:, :] += time_embed)
        """
        if dim_embed is None:
            cut_dimension = self.patch_size
        else:
            cut_dimension = dim_embed
        # first step will be match the time_embed to the number of channels
        temporal_embedding = self.temporal_embedding[:, 1:cut_dimension, :]
        # Add a new dimension to the time embedding
        # e.g. (batch, 62, 200) -> (batch, 1, 62, 200)
        temporal_embedding = temporal_embedding.unsqueeze(1)
        # Expand the time embedding to match the number of channels
        # or number of patches from
        temporal_embedding = temporal_embedding.expand(batch_size, num_ch, -1, -1)
        # Flatten the intermediate dimensions
        temporal_embedding = temporal_embedding.flatten(1, 2)
        return temporal_embedding

    def _adj_position_embedding(self, pos_embed_used, batch_size):
        """
        Adjust the dimensions of position embedding to match the
        number of patches.

        Parameters
        ----------
        pos_embed_used : torch.Tensor
            The position embedding to be adjusted.
        batch_size : int
            The number of batches.

        Returns
        -------
        pos_embed : torch.Tensor
            The adjusted position embedding
        """
        # [CLS] token has no position embedding
        pos_embed = pos_embed_used[:, 1:, :]
        # Adding a new dimension to the position embedding
        pos_embed = pos_embed.unsqueeze(2)
        # Need to expand the position embedding to match the number of
        # n_patches
        pos_embed = pos_embed.expand(batch_size, -1, self.patch_embed[0].n_patchs, -1)
        # Flatten the intermediate dimensions,
        # such as the number of patches and the "channels" dim
        pos_embed = pos_embed.flatten(1, 2)
        # Get the base position embedding
        # This is the position embedding for the [CLS] token
        base_pos = pos_embed[:, 0:1, :].expand(batch_size, -1, -1)
        # Concatenate the base position embedding with the
        # position embedding
        pos_embed = torch.cat((base_pos, pos_embed), dim=1)
        return pos_embed

class _SegmentPatch(nn.Module):
    """Segment and Patch for EEG data.

    Adapted Patch Embedding inspired in the Visual Transform approach
    to extract the learned segmentor, we expect get the input shape as:
    (Batch, Number of Channels, number of times points).

    We apply a 2D convolution with kernel size of (1, patch_size)
    and a stride of (1, patch_size).

    The results output shape will be:
    (Batch, Number of Channels, Number of patches, patch size).

    This way, we learned a convolution to segment the input shape.

    The number of patches is calculated as the number of samples divided
    by the patch size.

    Parameters:
    -----------
    n_times: int (default=2000)
        Number of temporal components of the input tensor.
    in_chans: int (default=1)
        number of electrods from the EEG signal
    emb_dim: int (default=200)
        Number of n_output to be used in the convolution, here,
        we used the same as patch_size.
    patch_size: int (default=200)
        Size of the patch, default is 1-seconds with 200Hz.
    Returns:
    --------
    x_patched: torch.Tensor
        Output tensor of shape (batch, n_chans, num_patches, emb_dim).
    """

    def __init__(
        self, n_times=2000, patch_size=200, n_chans=1, emb_dim=200, learned_patcher=True
    ):
        super().__init__()

        self.n_times = n_times
        self.patch_size = patch_size
        self.n_patchs = n_times // patch_size
        self.emb_dim = emb_dim
        self.n_chans = n_chans
        self.learned_patcher = learned_patcher

        self.patcher = nn.Conv1d(
            in_channels=1,
            out_channels=self.emb_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.adding_extra_dim = Rearrange(
            pattern="batch nchans temporal -> (batch nchans) 1 temporal"
        )

    def forward(self, x):
        """
        Using an 1D convolution to generate segments of EEG signal.

        Parameters:
        -----------
        X: Tensor
            [batch, n_chans, n_times]

        Returns:
        --------
        X_patch: Tensor
            [batch, n_chans, n_times//patch_size, patch_size]
        """
        batch_size, _, _ = x.shape
        # Input shape: [batch, n_chs, n_times]

        # First, rearrange input to treat the channel dimension 'n_chs' as
        # separate 'dimension' in batch for Conv1d
        # This requires reshaping x to have a height of 1 for each EEG sample.
        if self.learned_patcher:
            x = self.adding_extra_dim(x)

            # Apply the convolution along the temporal dimension
            # Conv2d output shape: [(batch*n_chs), emb_dim, n_patches]
            x = self.patcher(x)

            # Now, rearrange output to get back to a batch-first format,
            # combining embedded patches with channel information
            # Assuming you want [batch, n_chs, n_patches, emb_dim]
            # as output, which keeps channel information
            # This treats each patch embedding as a feature alongside channels
            x = rearrange(
                x,
                pattern="(batch nchans) embed npatchs -> batch nchans npatchs embed",
                batch=batch_size,
                nchans=self.n_chans,
            )
        else:
            x = x.view(
                batch_size,
                self.n_chans,
                self.n_times // self.patch_size,
                self.patch_size,
            )
        return x

