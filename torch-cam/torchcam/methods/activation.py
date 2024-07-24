class CAM(_CAM):
    r"""Implements a class activation map extractor as described in `"Learning Deep Features for Discriminative
    Localization" <https://arxiv.org/pdf/1512.04150.pdf>`_.

    The Class Activation Map (CAM) is defined for image classification models that have global pooling at the end
    of the visual feature extraction block. The localization map is computed as follows:

    .. math::
        L^{(c)}_{CAM}(x, y) = ReLU\Big(\sum\limits_k w_k^{(c)} A_k(x, y)\Big)

    where :math:`A_k(x, y)` is the activation of node :math:`k` in the target layer of the model at
    position :math:`(x, y)`,
    and :math:`w_k^{(c)}` is the weight corresponding to class :math:`c` for unit :math:`k` in the fully
    connected layer..

    >>> from torchvision.models import resnet18
    >>> from torchcam.methods import CAM
    >>> model = resnet18(pretrained=True).eval()
    >>> cam = CAM(model, 'layer4', 'fc')
    >>> with torch.no_grad(): out = model(input_tensor)
    >>> cam(class_idx=100)

    Args:
        model: input model
        target_layer: either the target layer itself or its name, or a list of those
        fc_layer: either the fully connected layer itself or its name
        input_shape: shape of the expected input tensor excluding the batch dimension
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[Union[Union[nn.Module, str], List[Union[nn.Module, str]]]] = None,
        fc_layer: Optional[Union[nn.Module, str]] = None,
        input_shape: Tuple[int, ...] = (3, 224, 224),
        **kwargs: Any,
    ) -> None:
        if isinstance(target_layer, list) and len(target_layer) > 1:
            raise ValueError("base CAM does not support multiple target layers")

        super().__init__(model, target_layer, input_shape, **kwargs)

        if isinstance(fc_layer, str):
            fc_name = fc_layer
        # Find the location of the module
        elif isinstance(fc_layer, nn.Module):
            fc_name = self._resolve_layer_name(fc_layer)
        # If the layer is not specified, try automatic resolution
        elif fc_layer is None:
            fc_name = locate_linear_layer(model)  # type: ignore[assignment]
            # Warn the user of the choice
            if isinstance(fc_name, str):
                logging.warning(f"no value was provided for `fc_layer`, thus set to '{fc_name}'.")
            else:
                raise ValueError("unable to resolve `fc_layer` automatically, please specify its value.")
        else:
            raise TypeError("invalid argument type for `fc_layer`")
        # Softmax weight
        self._fc_weights = self.submodule_dict[fc_name].weight.data
        # squeeze to accomodate replacement by Conv1x1
        if self._fc_weights.ndim > 2:
            self._fc_weights = self._fc_weights.view(*self._fc_weights.shape[:2])

    @torch.no_grad()
    def _get_weights(
        self,
        class_idx: Union[int, List[int]],
        *_: Any,
    ) -> List[Tensor]:
        """Computes the weight coefficients of the hooked activation maps."""
        # Take the FC weights of the target class
        if isinstance(class_idx, int):
            return [self._fc_weights[class_idx, :].unsqueeze(0)]
        else:
            return [self._fc_weights[class_idx, :]]

