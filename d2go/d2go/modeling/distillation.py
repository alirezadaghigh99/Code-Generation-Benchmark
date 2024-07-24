def _build_teacher(cfg) -> nn.Module:
    """Create teacher using config settings

    Supports torchscript or creating pytorch model using config.
    """
    _validate_teacher_config(cfg)
    if cfg.DISTILLATION.TEACHER.TYPE == "torchscript":
        with PathManager.open(cfg.DISTILLATION.TEACHER.TORCHSCRIPT_FNAME, "rb") as f:
            model = torch.jit.load(f)
    elif cfg.DISTILLATION.TEACHER.TYPE == "config":
        from d2go.runner import import_runner
        from d2go.setup import create_cfg_from_cli

        # teacher config may be set to cuda
        # if user wants to run teacher on cpu only machine by specifying teacher.device,
        # need to override device to cpu before building model
        if cfg.DISTILLATION.TEACHER.DEVICE:
            cfg.DISTILLATION.TEACHER.OVERWRITE_OPTS.extend(
                ["MODEL.DEVICE", cfg.DISTILLATION.TEACHER.DEVICE]
            )

        teacher_cfg = create_cfg_from_cli(
            cfg.DISTILLATION.TEACHER.CONFIG_FNAME,
            cfg.DISTILLATION.TEACHER.OVERWRITE_OPTS,
            cfg.DISTILLATION.TEACHER.RUNNER_NAME,
        )
        runner = import_runner(cfg.DISTILLATION.TEACHER.RUNNER_NAME)()
        model = runner.build_model(teacher_cfg, eval_only=True)
    elif cfg.DISTILLATION.TEACHER.TYPE == "no_teacher":
        model = nn.Identity()
    else:
        raise ValueError(f"Unexpected teacher type: {cfg.DISTILLATION.TEACHER.TYPE}")

    # move teacher to same device as student unless specified
    device = torch.device(cfg.DISTILLATION.TEACHER.DEVICE or cfg.MODEL.DEVICE)
    model = _set_device(model, device)
    model.eval()
    return model

def record_layers(model: nn.Module, layer_names: Set[str]) -> ModelOutput:
    """Save the outputs of layer_names in model

    Iterates over all named layers in model, applies cached layer to layers in
    layer_names. Returns dict which is used by the cached layers.
    """
    cache = {}
    for name, module in model.named_modules():
        if name in layer_names:
            dynamic_mixin(
                module,
                CachedLayer,
                init_dict={"label": name, "cache": cache},
            )
    return cache

class LayerLossMetadata:
    loss: nn.Module
    name: str
    layer0: str
    layer1: str

class DistillationModelingHook(mh.ModelingHook):
    """Wrapper hook that allows us to apply different distillation algorithms
    based on config

    This is meant to be used after creating a model:
        def build_model(cfg):
            model = d2_build_model(cfg)
            distillation_modeling_hook = DistillationModelingHook(cfg)
            d2go.modeling_hook.apply_modeling_hooks(model, distillation_modeling_hook)

    The udpated model will then be updated with a forward func that corresponds
    to the distillation method in the cfg as well as any new methods
    """

    def __init__(self, cfg):
        """
        Set the three major components
            distillation_algorithm_class => the distillation algorithm to be used, we
              only get the class as the apply() will mixin the class
            distillation_helper => user customization of the algorithm
            teacher => all distillation algorithms utilize an additional model to
              modify inputs
        """
        super().__init__(cfg)
        self.teacher = _build_teacher(cfg)
        self.distillation_algorithm_class = DISTILLATION_ALGORITHM_REGISTRY.get(
            cfg.DISTILLATION.ALGORITHM
        )
        self.distillation_helper = DISTILLATION_HELPER_REGISTRY.get(
            cfg.DISTILLATION.HELPER
        )(cfg, self.teacher)

    def apply(self, model: nn.Module) -> nn.Module:
        """Use dynamic mixin to apply the distillation class

        As opposed to wrapping the model, dynamic mixin allows us to override the
        model methods so that the model retains all existing attributes the user expects
        (e.g., if the user thinks their is an attr called model.my_attr then dynamic mixin
        retains that property). This has the advantage over directly overriding the model
        forward as we can still call the original model forward using super:

            old_model: MyModel
            new_model: MyDistillationClass = DistillationModelingHook(...).apply(old_model)

            class MyDistillationClass:
                def forward(self, ...):
                    # do some processing
                    ...
                    super().forward(...)  # call MyModel.forward
                    ...
        """
        logger.info("Applying distillation")
        dynamic_mixin(
            model,
            self.distillation_algorithm_class,
            init_dict={
                "distillation_helper": self.distillation_helper,
            },
        )
        return model

    def unapply(self, model: nn.Module) -> nn.Module:
        """Remove distillation class using dynamic mixin with saved original class"""
        remove_dynamic_mixin(model)
        return model

class DefaultLossCombiner:
    """Returns a weighted sum of the losses based on the name_weight

    name_weight is a dictionary indicating the name of the loss and the
    weight associated with that loss

    Example:
        name_weight = {"nll": 0.1, "kd": 0.9}
    """

    def __init__(self, name_weight: Dict[str, float]):
        self.name_weight = name_weight

    def __call__(self, losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = {}
        for k, v in losses.items():
            if k not in self.name_weight:
                raise ValueError(f"Unexpected weight in loss dict: {k}")
            output[k] = v * self.name_weight[k]
        return output

class NoopPseudoLabeler(PseudoLabeler):
    def label(self, x):
        return x

class BaseDistillationHelper:
    """Example of what distillation helper can provide

    Users should inherit this class and replace any functions with whatever they
    need in order to customize their distillation given a specific distililation
    algorithm (e.g., user wants to change the name of the label in the inputs).

    The distillation helper is an object passed to the distillation algorithm so
    any functionality in the helper can be accessed in the algorithm
    """

    def __init__(self, cfg: CN, teacher: nn.Module):
        self.cfg = cfg
        self.teacher = teacher

    def get_pseudo_labeler(self) -> PseudoLabeler:
        """
        pseudo_labeler should update the labels in batched_inputs with teacher model
        results

        This dummy psuedo_labeler returns the batched_inputs without modification
        """
        return NoopPseudoLabeler()

    def get_teacher(self) -> nn.Module:
        """Return a teacher that can be run by the algorithm"""
        return self.teacher

    def get_layer_losses(
        self, model: Optional[nn.Module] = None
    ) -> List[LayerLossMetadata]:
        """Return losses that are run on layers

        Layer parameters may be dependent on model parameters so option to pass
        in a model
        """
        return []

    def get_preprocess_student_input(self) -> Callable:
        """Return a function that allows user to modify the dataloader output
        before passing to the student

        The output of this function will be directly passed to the student model.
        Example use cases include:
            * dataloader returns a large image used by the teacher model but the
              student model needs a lower resolution version
            * dataloader returns both labeled and unlabeled data and the student
              requires labeled data
        """
        return lambda x: x

    def get_preprocess_teacher_input(self) -> Callable:
        """Return a function that allows user to modify dataloader output before
        passing to teacher

        The output of this function will be directly passed to the teacher model.
        """
        return lambda x: x

    def get_combine_losses(self) -> Callable:
        """Return a function that takes as input a dictionary of losses and
        modifies the loss as required

        The default trainer sums the losses at the end so typically this
        function is used to change the relative contribution of losses

        Example:
            def combine_losses(losses)
              alpha = 0.1
              losses["nll"] *= alpha
              losses["kd_loss"] *= (1 - alpha)
              return losses

            student_losses = {"nll": ...}
            student_losses.update({"kl_loss": ...})
            losses = combine_losses(student_losses)
        """
        return lambda x: x

    def get_preprocess_domain0_input(self) -> Callable:
        """Return a function that allows user to modify the dataloader output
        before passing to the model

        The output of this function will be directly passed to the model.
        Example use cases include:
          * dataloader returns a dictionary of real and synthetic images. use
          this function to return only the real data (domain0) to the model
        """
        return lambda x: x

    def get_preprocess_domain1_input(self) -> Callable:
        """Same as get_preprocess_domain0_input but returns domain1 inputs

        Example:
          * dataloader returns a dictionary of real and synthetic images. use
          this function to return only synthetic data (domain1) to the model
        """
        return lambda x: x

class ExampleDistillationHelper(BaseDistillationHelper):
    """
    This is an example of a user customizing distillation.

    We return a pseudo labeler that can be used with a specific project
    where the training input is a list of dicts with a label called target
    """

    def get_pseudo_labeler(self) -> PseudoLabeler:
        return RelabelTargetInBatch(self.teacher)

