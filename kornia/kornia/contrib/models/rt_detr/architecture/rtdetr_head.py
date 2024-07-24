class RTDETRHead(Module):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        in_channels: list[int],
        num_decoder_layers: int,
        num_heads: int = 8,
        num_decoder_points: int = 4,
        # num_levels: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        # TODO: verify this is correct
        self.num_levels = len(in_channels)

        # build the input projection layers
        self.input_proj = nn.ModuleList()
        for ch_in in in_channels:
            self.input_proj.append(ConvNormAct(ch_in, hidden_dim, 1, act="none"))
        # NOTE: might be missing some layers here ?
        # https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetr_pytorch/src/zoo/rtdetr/rtdetr_decoder.py#L403-L410

        # NOTE: need to be integrated with the TransformerDecoderLayer
        self.decoder_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    num_levels=len(in_channels),
                    num_points=num_decoder_points,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        self.decoder = TransformerDecoder(
            hidden_dim=hidden_dim, decoder_layers=self.decoder_layers, num_layers=num_decoder_layers
        )

        # denoising part
        self.denoising_class_embed = nn.Embedding(num_classes, hidden_dim)  # not used in evaluation

        # decoder embedding
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(num_decoder_layers)])
        self.dec_bbox_head = nn.ModuleList(
            [MLP(hidden_dim, hidden_dim, 4, num_layers=3) for _ in range(num_decoder_layers)]
        )

    def forward(self, feats: Tensor) -> tuple[Tensor, Tensor]:
        # input projection and embedding
        memory, spatial_shapes, level_start_index = self._get_encoder_input(feats)

        # prepare denoising training
        denoising_class, denoising_bbox_unact, attn_mask = None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = self._get_decoder_input(
            memory, spatial_shapes, denoising_class, denoising_bbox_unact
        )

        # decoder
        out_bboxes, out_logits = self.decoder.forward(
            target,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )

        return out_logits[-1], out_bboxes[-1]

    def _get_encoder_input(self, feats: Tensor) -> tuple[Tensor, list[tuple[int, int]], list[int]]:
        # get projection features
        proj_feats: list[Tensor] = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten_list: list[Tensor] = []
        spatial_shapes: list[tuple[int, int]] = []
        level_start_index: list[int] = [0]

        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten_list.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append((h, w))
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten: Tensor = concatenate(feat_flatten_list, 1)

        level_start_index.pop()
        return feat_flatten, spatial_shapes, level_start_index

    def _get_decoder_input(
        self,
        memory: Tensor,
        spatial_shapes: list[tuple[int, int]],
        denoising_class: Optional[Tensor] = None,
        denoising_bbox_unact: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # prepare input for decoder
        # TODO: cache anchors and valid_mask as buffers
        anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device, dtype=memory.dtype)

        # memory = torch.where(valid_mask, memory, 0)
        memory = valid_mask.to(memory) * memory  # TODO fix type error for onnx export

        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.num_queries, dim=1)

        reference_points_unact = enc_outputs_coord_unact.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact.shape[-1])
        )

        enc_topk_bboxes = torch.sigmoid(reference_points_unact)
        if denoising_bbox_unact is not None:
            reference_points_unact = torch.concat([denoising_bbox_unact, reference_points_unact], 1)

        enc_topk_logits = enc_outputs_class.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1])
        )

        # extract region features
        target = output_memory.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))

        if denoising_class is not None:
            target = torch.concat([denoising_class, target], 1)

        return target.detach(), reference_points_unact.detach(), enc_topk_bboxes, enc_topk_logits

    @staticmethod
    def _generate_anchors(
        spatial_shapes: list[tuple[int, int]],
        grid_size: float = 0.05,
        eps: float = 0.01,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> tuple[Tensor, Tensor]:
        """Generate anchors for RT-DETR.

        Args:
            spatial_shapes: shape (width, height) of the feature maps
            grid_size: size of the grid
            eps: specify the minimum and maximum size of the anchors
            device: device to place the anchors
            dtype: data type for the anchors

        Returns:
            logit of anchors and mask
        """
        # TODO: might make this (or some parts of it) into a separate reusable function
        anchors_list: list[Tensor] = []

        for i, (h, w) in enumerate(spatial_shapes):
            # TODO: fix later kornia.utils.create_meshgrid()
            grid_y, grid_x = torch_meshgrid(
                [torch.arange(h, device=device, dtype=dtype), torch.arange(w, device=device, dtype=dtype)],
                indexing="ij",
            )
            grid_xy = torch.stack([grid_x, grid_y], -1)  # HxWx2

            # this satisfies onnx export
            wh = torch.empty(2, device=device, dtype=dtype)
            wh[0] = w
            wh[1] = h

            grid_xy = (grid_xy + 0.5) / wh  # normalize to [0, 1]
            grid_wh = torch.ones_like(grid_xy) * grid_size * (2.0**i)
            anchors_list.append(concatenate([grid_xy, grid_wh], -1).reshape(-1, h * w, 4))

        anchors = concatenate(anchors_list, 1)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))  # anchors.logit() fails ONNX export

        inf_t = torch.empty(1, device=device, dtype=dtype)
        inf_t[0] = float("inf")

        anchors = torch.where(valid_mask, anchors, inf_t)

        return anchors, valid_mask

