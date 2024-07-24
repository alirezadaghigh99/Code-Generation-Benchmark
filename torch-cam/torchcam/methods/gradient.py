class SmoothGradCAMpp(_GradCAM):
    r"""Implements a class activation map extractor as described in `"Smooth Grad-CAM++: An Enhanced Inference Level
    Visualization Technique for Deep Convolutional Neural Network Models" <https://arxiv.org/pdf/1908.01224.pdf>`_
    with a personal correction to the paper (alpha coefficient numerator).

    The localization map is computed as follows:

    .. math::
        L^{(c)}_{Smooth Grad-CAM++}(x, y) = \sum\limits_k w_k^{(c)} A_k(x, y)

    with the coefficient :math:`w_k^{(c)}` being defined as:

    .. math::
        w_k^{(c)} = \sum\limits_{i=1}^H \sum\limits_{j=1}^W \alpha_k^{(c)}(i, j) \cdot
        ReLU\Big(\frac{\partial Y^{(c)}}{\partial A_k(i, j)}\Big)

    where :math:`A_k(x, y)` is the activation of node :math:`k` in the target layer of the model at
    position :math:`(x, y)`,
    :math:`Y^{(c)}` is the model output score for class :math:`c` before softmax,
    and :math:`\alpha_k^{(c)}(i, j)` being defined as:

    .. math::
        \alpha_k^{(c)}(i, j)
        = \frac{\frac{\partial^2 Y^{(c)}}{(\partial A_k(i,j))^2}}{2 \cdot
        \frac{\partial^2 Y^{(c)}}{(\partial A_k(i,j))^2} + \sum\limits_{a,b} A_k (a,b) \cdot
        \frac{\partial^3 Y^{(c)}}{(\partial A_k(i,j))^3}}
        = \frac{\frac{1}{n} \sum\limits_{m=1}^n D^{(c, 2)}_k(i, j)}{
        \frac{2}{n} \sum\limits_{m=1}^n D^{(c, 2)}_k(i, j) + \sum\limits_{a,b} A_k (a,b) \cdot
        \frac{1}{n} \sum\limits_{m=1}^n D^{(c, 3)}_k(i, j)}

    if :math:`\frac{\partial Y^{(c)}}{\partial A_k(i, j)} = 1` else :math:`0`. Here :math:`D^{(c, p)}_k(i, j)`
    refers to the p-th partial derivative of the class score of class :math:`c` relatively to the activation in layer
    :math:`k` at position :math:`(i, j)`, and :math:`n` is the number of samples used to get the gradient estimate.

    Please note the difference in the numerator of :math:`\alpha_k^{(c)}(i, j)`,
    which is actually :math:`\frac{1}{n} \sum\limits_{k=1}^n D^{(c, 1)}_k(i,j)` in the paper.

    >>> from torchvision.models import resnet18
    >>> from torchcam.methods import SmoothGradCAMpp
    >>> model = resnet18(pretrained=True).eval()
    >>> cam = SmoothGradCAMpp(model, 'layer4')
    >>> scores = model(input_tensor)
    >>> cam(class_idx=100)

    Args:
        model: input model
        target_layer: either the target layer itself or its name, or a list of those
        num_samples: number of samples to use for smoothing
        std: standard deviation of the noise
        input_shape: shape of the expected input tensor excluding the batch dimension
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[Union[Union[nn.Module, str], List[Union[nn.Module, str]]]] = None,
        num_samples: int = 4,
        std: float = 0.3,
        input_shape: Tuple[int, ...] = (3, 224, 224),
        **kwargs: Any,
    ) -> None:
        super().__init__(model, target_layer, input_shape, **kwargs)
        # Model scores is not used by the extractor
        self._score_used = False

        # Input hook
        self.hook_handles.append(model.register_forward_pre_hook(self._store_input))  # type: ignore[arg-type]
        # Noise distribution
        self.num_samples = num_samples
        self.std = std
        self._distrib = torch.distributions.normal.Normal(0, self.std)
        # Specific input hook updater
        self._ihook_enabled = True

    def _store_input(self, _: nn.Module, _input: Tensor) -> None:
        """Store model input tensor."""
        if self._ihook_enabled:
            self._input = _input[0].data.clone()

    def _get_weights(
        self,
        class_idx: Union[int, List[int]],
        _: Union[Tensor, None] = None,
        eps: float = 1e-8,
        **kwargs: Any,
    ) -> List[Tensor]:
        """Computes the weight coefficients of the hooked activation maps."""
        # Disable input update
        self._ihook_enabled = False
        # Keep initial activation
        self.hook_a: List[Tensor]  # type: ignore[assignment]
        self.hook_g: List[Tensor]  # type: ignore[assignment]
        init_fmap = [act.clone() for act in self.hook_a]
        # Initialize our gradient estimates
        grad_2 = [torch.zeros_like(act) for act in self.hook_a]
        grad_3 = [torch.zeros_like(act) for act in self.hook_a]
        # Perform the operations N times
        for _idx in range(self.num_samples):
            # Add noise
            noisy_input = self._input + self._distrib.sample(self._input.size()).to(device=self._input.device)
            noisy_input.requires_grad_(True)
            # Forward & Backward
            out = self.model(noisy_input)
            self.model.zero_grad()
            self._backprop(out, class_idx, **kwargs)

            # Sum partial derivatives
            grad_2 = [g2.add_(grad.pow(2)) for g2, grad in zip(grad_2, self.hook_g)]
            grad_3 = [g3.add_(grad.pow(3)) for g3, grad in zip(grad_3, self.hook_g)]

        # Reenable input update
        self._ihook_enabled = True

        # Average the gradient estimates
        grad_2 = [g2.div_(self.num_samples) for g2 in grad_2]
        grad_3 = [g3.div_(self.num_samples) for g3 in grad_3]

        # Alpha coefficient for each pixel
        spatial_dims = self.hook_a[0].ndim - 2
        alpha = [
            g2 / (2 * g2 + (g3 * act).flatten(2).sum(-1)[(...,) + (None,) * spatial_dims] + eps)
            for g2, g3, act in zip(grad_2, grad_3, init_fmap)
        ]

        # Apply pixel coefficient in each weight
        return [a.mul_(torch.relu(grad)).flatten(2).sum(-1) for a, grad in zip(alpha, self.hook_g)]

    def extra_repr(self) -> str:
        return f"target_layer={self.target_names}, num_samples={self.num_samples}, std={self.std}"

