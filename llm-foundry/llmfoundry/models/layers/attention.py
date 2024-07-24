class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA) is a generalization of Multi-head (MHA).

    and Multi-query attention (MQA).

    This allows the user to set a variable of number of kv_n_heads, rather than
    just n_heads or 1, as in MHA and MQA. Using torch attention
    implementation enables user to also use additive bias.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        kv_n_heads: int,
        attn_impl: str = 'flash',
        clip_qkv: Optional[float] = None,
        qk_ln: bool = False,
        qk_gn: bool = False,
        softmax_scale: Optional[float] = None,
        attn_pdrop: float = 0.0,
        norm_type: str = 'low_precision_layernorm',
        fc_type: Optional[dict[str, Any]] = None,
        device: Optional[str] = None,
        bias: bool = True,
        sliding_window_size: int = -1,
        reuse_kv_layer_idx: Optional[int] = None,
    ):
        super().__init__()

        self.attn_impl = attn_impl
        self.clip_qkv = clip_qkv
        self.qk_ln = qk_ln
        self.qk_gn = qk_gn

        self.d_model = d_model
        self.n_heads = n_heads
        self.kv_n_heads = kv_n_heads
        self.sliding_window_size = sliding_window_size
        self.reuse_kv_layer_idx = reuse_kv_layer_idx

        self.head_dim = d_model // n_heads

        # Usually, fc_type dict should be passed in through MPTBlock's __init__ function.
        if fc_type is None:
            fc_type = copy.deepcopy(fc_type_defaults)
            fc_type['bias'] = bias
            fc_type['device'] = device
        fc_type_name = fc_type['name']

        if self.kv_n_heads <= 0:
            raise ValueError('kv_n_heads should be greater than zero.')

        if self.kv_n_heads > self.n_heads:
            raise ValueError(
                'The number of KV heads should be less than or equal to Q heads.',
            )

        if self.n_heads % self.kv_n_heads != 0:
            raise ValueError(
                'Each Q head should get the same number of KV heads, so n_heads must be divisible by kv_n_heads.',
            )
        if qk_ln and qk_gn:
            raise ValueError('Only one of qk_ln and qk_gn can be set to True.')

        self.softmax_scale = softmax_scale
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.d_model / self.n_heads)
        self.attn_dropout_p = attn_pdrop

        if self.reuse_kv_layer_idx is None:
            self.Wqkv = build_fc(
                name=fc_type_name,
                in_features=self.d_model,
                out_features=self.d_model + 2 * self.kv_n_heads * self.head_dim,
                fc_kwargs=fc_type,
            )
            # for param init fn; enables shape based init of fused layers
            fuse_splits = [
                i * self.head_dim
                for i in range(1, self.n_heads + 2 * self.kv_n_heads)
            ]
            self.Wqkv._fused = (0, fuse_splits)
        else:
            self.Wq = build_fc(
                name=fc_type_name,
                in_features=self.d_model,
                out_features=self.d_model,
                fc_kwargs=fc_type,
            )
            # for param init fn; enables shape based init of fused layers
            fuse_splits = [i * self.head_dim for i in range(1, self.n_heads)]
            self.Wq._fused = (0, fuse_splits)

        if self.qk_ln or self.qk_gn:
            norm_size = self.head_dim if qk_gn else d_model
            self.q_ln = build_norm(
                name=norm_type.lower(),
                normalized_shape=norm_size,
                device=device,
            )
            if self.reuse_kv_layer_idx is None:
                if qk_ln:
                    norm_size = self.head_dim * kv_n_heads
                self.k_ln = build_norm(
                    name=norm_type.lower(),
                    normalized_shape=norm_size,
                    device=device,
                )

        self.attn_fn = attention_implementations.get(self.attn_impl)

        self.out_proj = build_fc(
            name=fc_type_name,
            in_features=self.d_model,
            out_features=self.d_model,
            fc_kwargs=fc_type,
        )
        self.out_proj._is_residual = True

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attn_bias: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb_w_meta_info: Optional[dict] = None,
        is_causal: bool = True,
        needs_weights: bool = False,
        alibi_slopes: Optional[torch.Tensor] = None,
        flash_attn_padding_info: Optional[dict[str, torch.Tensor]] = None,
        prev_layer_key_value: Optional[Tuple[torch.Tensor,
                                             torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[
        torch.Tensor, torch.Tensor]]]:
        extra_kwargs = {}
        if prev_layer_key_value is not None:
            extra_kwargs['prev_layer_key_value'] = prev_layer_key_value
        query, key, value = self.get_qkv(x, **extra_kwargs)

        if rotary_emb_w_meta_info is not None:
            query, key, value = self._apply_rotary_embeddings(
                rotary_emb_w_meta_info,
                query,
                key,
                value,
            )

        extra_attn_kwargs = self.get_implementation_specific_args(
            attention_mask,
            alibi_slopes,
            flash_attn_padding_info,
        )

        context, attn_weights, past_key_value = self.attn_fn(
            query,
            key,
            value,
            n_heads=self.n_heads,
            kv_n_heads=self.kv_n_heads,
            past_key_value=past_key_value,
            softmax_scale=self.softmax_scale,
            attn_bias=attn_bias,
            is_causal=is_causal,
            dropout_p=self.attn_dropout_p,
            training=self.training,
            needs_weights=needs_weights,
            **extra_attn_kwargs,
        )

        return self.out_proj(context), attn_weights, past_key_value

    def get_qkv(
        self,
        x: torch.Tensor,
        prev_layer_key_value: Optional[Tuple[torch.Tensor,
                                             torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes and returns the query, key, and value tensors.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            query (torch.Tensor): The query tensor.
            key (torch.Tensor): The key tensor.
            value (torch.Tensor): The value tensor.
        """
        if self.reuse_kv_layer_idx is not None:
            if prev_layer_key_value is None:
                raise ValueError(
                    'prev_layer_key_value is None, cannot reuse_prev_layer_kv.',
                )
            key, value = prev_layer_key_value

            query = self.Wq(x)
            if self.clip_qkv:
                query = query.clamp(min=-self.clip_qkv, max=self.clip_qkv)

            if self.qk_ln or self.qk_gn:
                # Applying layernorm to qk
                q_shape = query.shape
                if self.qk_gn:
                    b, s = query.shape[:2]
                    query = query.view(b, s, self.n_heads, -1)
                dtype = query.dtype
                query = self.q_ln(query).to(dtype).view(q_shape)
            return query, key, value

        qkv = self.Wqkv(x)

        if self.clip_qkv:
            qkv = qkv.clamp(min=-self.clip_qkv, max=self.clip_qkv)

        query, key, value = qkv.split(
            [
                self.d_model,
                self.kv_n_heads * self.head_dim,
                self.kv_n_heads * self.head_dim,
            ],
            dim=2,
        )

        if self.qk_ln or self.qk_gn:
            # Applying layernorm to qk
            q_shape, k_shape = query.shape, key.shape
            if self.qk_gn:
                b, s = query.shape[:2]
                query = query.view(b, s, self.n_heads, -1)
                key = key.view(b, s, self.kv_n_heads, -1)
            dtype = query.dtype
            query = self.q_ln(query).to(dtype).view(q_shape)
            key = self.k_ln(key).to(dtype).view(k_shape)

        return query, key, value

    def _apply_rotary_embeddings(
        self,
        rotary_emb_w_meta_info: Dict[str, Any],
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.reuse_kv_layer_idx is not None:
            orig_key, orig_value = key, value
            key, value = torch.empty_like(key), torch.empty_like(value)

        rotary_emb = rotary_emb_w_meta_info['rotary_emb']
        seq_len = rotary_emb_w_meta_info['seq_len']
        offset_info = rotary_emb_w_meta_info['offset_info']
        bsz, seqlen = query.shape[:2]
        query = query.view(bsz, seqlen, -1, self.head_dim)
        key = key.view(bsz, seqlen, -1, self.head_dim)

        if rotary_emb_w_meta_info['impl'] == 'dail':
            value = value.view(bsz, seqlen, -1, self.head_dim)

            kv = torch.stack([key, value], dim=2)
            # Note: Rotates in place (https://github.com/Dao-AILab/flash-attention/blob/320fb59487658f033f56711efd3d61b7c7a6f8f3/flash_attn/layers/rotary.py#L429)
            query, kv = rotary_emb(
                query,
                kv,
                seqlen_offset=offset_info,
                max_seqlen=seq_len,
            )
            [key, value] = torch.unbind(kv, dim=2)

            value = value.view(bsz, seqlen, -1)
        elif rotary_emb_w_meta_info['impl'] == 'hf':
            if is_transformers_version_gte('4.38'):
                (cos, sin) = rotary_emb(
                    x=value,
                    position_ids=offset_info,
                )
            else:
                (cos, sin) = rotary_emb(x=value, seq_len=seq_len)
            if is_transformers_version_gte('4.38'):
                # In the following lines we move the cos and sin tensors to the same devices as query. These operations should be no-ops during training.
                # This is done to fix pipeline parallel generation using hf.generate. Please see this comment for details: https://github.com/mosaicml/llm-foundry/pull/1332#issue-2386827204
                cos = cos.to(query.device)
                sin = sin.to(query.device)
                query, key = apply_rotary_pos_emb(
                    q=query,
                    k=key,
                    cos=cos,
                    sin=sin,
                    position_ids=None,
                    unsqueeze_dim=2,
                )
            elif is_transformers_version_gte('4.36'):
                query, key = apply_rotary_pos_emb(
                    q=query,
                    k=key,
                    cos=cos,
                    sin=sin,
                    position_ids=offset_info,
                    unsqueeze_dim=2,
                )
            else:
                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
                query, key = apply_rotary_pos_emb(
                    q=query,
                    k=key,
                    cos=cos,
                    sin=sin,
                    position_ids=offset_info,
                )
                query = query.transpose(1, 2)
                key = key.transpose(1, 2)

        query = query.view(bsz, seqlen, -1)
        key = key.view(bsz, seqlen, -1)
        if self.reuse_kv_layer_idx is not None:
            return query, orig_key, orig_value  # type: ignore
        return query, key, value

    def get_implementation_specific_args(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
        flash_attn_padding_info: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, Any]:
        """Returns attention implementation specific args.

        Args:
            attention_mask (Optional[torch.Tensor]): The attention mask.
            alibi_slopes (Optional[torch.Tensor]): The alibi slopes.
            flash_attn_padding_info (Optional[dict[str, torch.Tensor]]): The padding information, only required for flash attention.

        Returns:
            extra_attn_kwargs (dict[str, Any]): Implementation specific args.
        """
        if self.attn_impl == 'flash':
            extra_attn_kwargs = {
                'should_repeat_kv_for_gqa': not is_flash_v2_installed(),
                'sliding_window_size': self.sliding_window_size,
                'alibi_slopes': alibi_slopes,
                'flash_attn_padding_info': flash_attn_padding_info,
                'key_padding_mask': None,
            }
        else:
            extra_attn_kwargs = {'key_padding_mask': attention_mask}
        return extra_attn_kwargs

class MultiheadAttention(GroupedQueryAttention):
    """Multi-head self attention.

    Using torch attention implementation enables user to also use additive bias.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_impl: str = 'flash',
        clip_qkv: Optional[float] = None,
        qk_ln: bool = False,
        qk_gn: bool = False,
        softmax_scale: Optional[float] = None,
        attn_pdrop: float = 0.0,
        norm_type: str = 'low_precision_layernorm',
        fc_type: Optional[dict[str, Any]] = None,
        device: Optional[str] = None,
        bias: bool = True,
        sliding_window_size: int = -1,
        reuse_kv_layer_idx: Optional[int] = None,
    ):
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            kv_n_heads=n_heads,  # for MHA, same # heads as kv groups
            attn_impl=attn_impl,
            clip_qkv=clip_qkv,
            qk_ln=qk_ln,
            qk_gn=qk_gn,
            softmax_scale=softmax_scale,
            attn_pdrop=attn_pdrop,
            norm_type=norm_type,
            fc_type=fc_type,
            device=device,
            bias=bias,
            sliding_window_size=sliding_window_size,
            reuse_kv_layer_idx=reuse_kv_layer_idx,
        )

