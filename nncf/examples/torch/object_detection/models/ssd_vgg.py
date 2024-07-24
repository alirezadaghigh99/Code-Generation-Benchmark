class SSD_VGG(nn.Module):
    def __init__(self, cfg, size, num_classes, batch_norm=False):
        super().__init__()
        self.config = cfg
        self.num_classes = num_classes
        self.size = size
        self.enable_batchmorm = batch_norm

        base_layers, base_outs, base_feats = build_vgg_ssd_layers(
            BASE_NUM_OUTPUTS[size], BASE_OUTPUT_INDICES[size], batch_norm=batch_norm
        )
        extra_layers, extra_outs, extra_feats = build_vgg_ssd_extra(
            EXTRAS_NUM_OUTPUTS[size], EXTRA_OUTPUT_INDICES[size], batch_norm=batch_norm
        )
        self.basenet = MultiOutputSequential(base_outs, base_layers)
        self.extras = MultiOutputSequential(extra_outs, extra_layers)

        self.detection_head = SSDDetectionOutput(base_feats + extra_feats, num_classes, cfg)
        self.L2Norm = L2Norm(512, 20, 1e-10)

    def forward(self, x):
        img_tensor = x[0].clone().unsqueeze(0)

        sources, x = self.basenet(x)
        sources[0] = self.L2Norm(sources[0])

        extra_sources, x = self.extras(x)

        return self.detection_head(sources + extra_sources, img_tensor)

    def load_weights(self, base_file):
        _, ext = os.path.splitext(base_file)
        if ext in [".pkl", ".pth"]:
            logger.debug("Loading weights into state dict...")
            #
            # ** WARNING: torch.load functionality uses Python's pickling facilities that
            # may be used to perform arbitrary code execution during unpickling. Only load the data you
            # trust.
            #
            self.load_state_dict(
                torch.load(base_file, map_location=lambda storage, loc: storage, pickle_module=restricted_pickle_module)
            )
            logger.debug("Finished!")
        else:
            logger.error("Sorry only .pth and .pkl files supported.")

