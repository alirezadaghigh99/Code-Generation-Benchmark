class MILANViT(MAEViT):
    """Vision Transformer for MILAN pre-training.

    Implementation of the encoder for `MILAN: Masked Image Pretraining on
    Language Assisted Representation <https://arxiv.org/abs/2208.06049>`_.

    This module inherits from MAEViT and only overrides the forward function
    and replace random masking with attention masking.
    """

    def attention_masking(
        self, x: torch.Tensor, mask_ratio: float, importance: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate attention mask for MILAN.

        This is what is different from MAEViT, which uses random masking.
        Attention masking generates attention mask for MILAN, according to
        importance. The higher the importance, the more likely the patch is
        kept.

        Args:
            x (torch.Tensor): Input images, which is of shape B x L x C.
            mask_ratio (float): The ratio of patches to be masked.
            importance (torch.Tensor): Importance of each patch, which is of
                shape B x L.

        Returns:
            Tuple[torch.Tensor, ...]:

            - ``x_masked``: masked image
            - ``ids_restore``: the ids to restore original image
            - ``ids_keep``: ids of the kept patches
            - ``ids_dump``: ids of the removed patches
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = importance.to(x.device)  # large is keep, small is remove

        # sort noise for each sample
        ids_shuffle = torch.multinomial(noise, L, replacement=False)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_dump = ids_shuffle[:, len_keep:]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, ids_restore, ids_keep, ids_dump

    def forward(
        self,
        x: torch.Tensor,
        importance: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate features for masked images.

        The function supports two kind of forward behaviors. If the
        ``importance`` is ``None``, the function generates mask and masks some
        patches randomly and get the hidden features for visible patches. The
        mask is generated by importance. The higher the importance, the more
        likely the patch is kept. The importance is calculated by CLIP.
        The higher the CLIP score, the more likely the patch is kept. The CLIP
        score is calculated by cross attention between the class token and all
        other tokens from the last layer.
        If the ``importance`` is ``torch.Tensor``, the forward function will
        call ``super().forward()``, which extract features from images without
        mask.

        Args:
            x (torch.Tensor): Input images, which is of shape B x C x H x W.
            importance (torch.Tensor, optional): Importance of each patch,
                which is of shape B x L.

        Returns:
            Tuple[torch.Tensor, ...]: masked image, the ids to restore original
            image, ids of the kept patches, ids of the removed patches.

            - ``x`` (torch.Tensor): hidden features, which is of shape
              B x (L * mask_ratio) x C.
            - ``ids_restore`` (torch.Tensor): ids to restore original image.
            - ``ids_keep`` (torch.Tensor): ids of the kept patches.
            - ``ids_dump`` (torch.Tensor): ids of the removed patches.
        """
        if importance is None:
            return super(MAEViT, self).forward(x)

        else:
            B = x.shape[0]
            x = self.patch_embed(x)[0]
            # add pos embed w/o cls token
            x = x + self.pos_embed[:, 1:, :]

            # masking: length -> length * mask_ratio
            x, ids_restore, ids_keep, ids_dump = self.attention_masking(
                x, self.mask_ratio, importance)

            # append cls token
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

            for _, layer in enumerate(self.layers):
                x = layer(x)
            # Use final norm
            x = self.norm1(x)

            return x, ids_restore, ids_keep, ids_dump
