class ContrastiveLossClip(BaseMultiModalImageTextCriteria):
    """Compute contrastive loss between image and text pairs.

    Args:
        opts: command-line arguments
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)
        # need to set these default to prevent tests for failing
        self.rank = getattr(opts, "ddp.rank", 0)
        self.use_distributed = getattr(opts, "ddp.use_distributed", False)
        self.device = getattr(opts, "dev.device", torch.device("cpu"))

    def _forward_clip(
        self, prediction: Dict[str, Tensor], *args, **kwargs
    ) -> Dict[str, Tensor]:
        """
        Computes the contrast loss between image and text representations

        Args:
            prediction: A mapping of the form (string: Tensor). image and text are mandatory keys

        Shape:
            prediction["image"]: Shape is [N, d]
            prediction["text"]: Shape is [N, d]
            where N and d are batch size and feature dimensions, respectively.

        Returns:
            The output dictionary contains four keys (total_loss, image_loss, text_loss, logit_scale) and scalar
            loss value for each of these keys. total_loss is sum of image_loss and text_loss.
        """

        if not {"image", "text"}.issubset(prediction.keys()):
            logger.error(
                f"image and text are mandatory keys for {self.__class__.__name__}."
            )

        image_features = prediction.pop("image")
        text_features = prediction.pop("text")
        logit_scale = prediction.pop("logit_scale", 1.0)

        if image_features is None:
            logger.error(f"Image features can't be None in {self.__class__.__name__}")
        if text_features is None:
            logger.error(f"Text features can't be None in {self.__class__.__name__}")

        # Aggregate image and text features from all GPUs
        gathered_image_features, gathered_text_features = gather_features(
            image_features=image_features,
            text_features=text_features,
            use_distributed=self.use_distributed,
        )

        # compute logits
        # [N, d] x [G x d]^T --> [N, G], where G is global batch size
        logits_per_image = logit_scale * (
            image_features @ gathered_text_features.transpose(0, 1)
        )
        # [N, d] x [G, d]^T --> [N, G]
        logits_per_text = logit_scale * (
            text_features @ gathered_image_features.transpose(0, 1)
        )

        # generate labels
        num_logits = logits_per_image.shape[0]
        contrastive_labels = torch.arange(
            num_logits, device=logits_per_image.device, dtype=torch.long
        )

        # shift the labels by rank id
        contrastive_labels = contrastive_labels + (num_logits * self.rank)

        # compute cross entropy loss
        text_loss = F.cross_entropy(logits_per_text, contrastive_labels) * 0.5
        image_loss = F.cross_entropy(logits_per_image, contrastive_labels) * 0.5
        total_loss = image_loss + text_loss
        return {
            "total_loss": total_loss,
            "image_loss": image_loss,
            "text_loss": text_loss,
            "logit_scale": logit_scale,
        }

    def forward(
        self,
        input_sample: Any,
        prediction: Dict[str, Tensor],
        *args,
        **kwargs,
    ) -> Dict:
        """
        Computes contrastive loss between image and text representations, optionally with neural aug

        Args:
            input_sample: Input to the model.
            prediction: A mapping of the form (string: Tensor). image and text are mandatory keys.

        Shape:
            input_sample: This loss function does not care about this argument.
            prediction["image"]: Shape is [N, d]
            prediction["text"]: Shape is [N, d]

            where N is the local batch size and d is the feature dimension.

        Returns:
            The output dictionary contains four keys (total_loss, image_loss, text_loss, logit_scale) and scalar
            loss value for each of these keys. total_loss is sum of image_loss and text_loss.
        """

        if not self.training:
            # we typically compute zero-shot logits for monitoring the val perf.
            # Therefore, we return 0 for loss during validation.

            # Note: In future, we may compute validation loss (depending on use case)
            return {
                "total_loss": torch.tensor(0.0, dtype=torch.float, device=self.device)
            }

        clip_loss_dict = self._forward_clip(prediction=prediction)
        return clip_loss_dict

