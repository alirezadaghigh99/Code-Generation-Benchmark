class IntermediateLayerGetter(Model):
    """Implements an intermediate layer getter

    >>> from tensorflow.keras.applications import ResNet50
    >>> from doctr.models import IntermediateLayerGetter
    >>> target_layers = ["conv2_block3_out", "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    >>> feat_extractor = IntermediateLayerGetter(ResNet50(include_top=False, pooling=False), target_layers)

    Args:
    ----
        model: the model to extract feature maps from
        layer_names: the list of layers to retrieve the feature map from
    """

    def __init__(self, model: Model, layer_names: List[str]) -> None:
        intermediate_fmaps = [model.get_layer(layer_name).get_output_at(0) for layer_name in layer_names]
        super().__init__(model.input, outputs=intermediate_fmaps)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

