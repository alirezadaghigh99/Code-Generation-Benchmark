class CLExperience:
    """
    Base Experience.

    Experiences have an index which track the experience's position
    inside the stream for evaluation purposes.
    """

    def __init__(
        self: TCLExperience,
        current_experience: int,
        origin_stream: "Optional[CLStream[TCLExperience]]",
    ):
        super().__init__()
        self._current_experience: int = current_experience
        """Experience identifier (the position in the origin_stream)."""

        self._origin_stream: "Optional[CLStream[TCLExperience]]" = origin_stream
        """Stream containing the experience."""

        self._exp_mode: ExperienceMode = ExperienceMode.LOGGING
        # used to block access to private info (e.g. task labels,
        # past experiences).

        self._unmask_context_depth = 0

        self._as_attributes("_current_experience")

    @property
    def current_experience(self) -> int:
        curr_exp = self._current_experience
        CLExperience._check_unset_attribute("current_experience", curr_exp)
        return curr_exp

    @current_experience.setter
    def current_experience(self, id: int):
        self._current_experience = id

    @property
    def origin_stream(self: TCLExperience) -> "CLStream[TCLExperience]":
        orig_stream = self._origin_stream
        CLExperience._check_unset_attribute("origin_stream", orig_stream)
        return orig_stream

    @origin_stream.setter
    def origin_stream(self: TCLExperience, stream: "CLStream[TCLExperience]"):
        self._origin_stream = stream

    @contextmanager
    def no_attribute_masking(self):
        try:
            self._unmask_context_depth += 1
            assert self._unmask_context_depth > 0
            yield
        finally:
            self._unmask_context_depth -= 1
            assert self._unmask_context_depth >= 0

    @property
    def are_attributes_masked(self) -> bool:
        return self._unmask_context_depth == 0

    def __getattribute__(self, item):
        """Custom getattribute.

        Check that ExperienceAttribute are available in train/eval mode.
        """
        v = super().__getattribute__(item)

        if isinstance(v, ExperienceAttribute):
            if not self.are_attributes_masked:
                return v.value
            elif self._exp_mode == ExperienceMode.TRAIN and v.use_in_train:
                return v.value
            elif self._exp_mode == ExperienceMode.EVAL and v.use_in_eval:
                return v.value
            elif self._exp_mode == ExperienceMode.LOGGING:
                return v.value
            else:
                mode = "train" if self._exp_mode == ExperienceMode.TRAIN else "eval"
                se = (
                    f"Attribute {item} is not available for the experience "
                    f"in {mode} mode."
                )
                raise MaskedAttributeError(se)
        else:
            return v

    def __setattr__(self, name, value):
        try:
            v = self.__dict__[name]
        except KeyError:
            return super().__setattr__(name, value)

        if isinstance(v, ExperienceAttribute):
            if isinstance(value, ExperienceAttribute):
                super().__setattr__(name, value)
            else:
                v.value = value
        else:
            return super().__setattr__(name, value)

    def _as_attributes(self, *fields: str, use_in_train=False, use_in_eval=False):
        """
        Internal method used to transform plain object fields to
        ExperienceAttribute(s).

        This is needed to ensure that static type checkers will not consider
        those fields as being of type "ExperienceAttribute", as this may be
        detrimental on the user experience.
        """
        for field in fields:
            v = super().__getattribute__(field)
            if isinstance(v, ExperienceAttribute):
                if v.use_in_train != use_in_train:
                    raise RuntimeError(
                        f"Experience attribute {field} redefined with "
                        f"incongruent use_in_train field. Was "
                        f"{v.use_in_train}, overridden with {use_in_train}."
                    )

                if v.use_in_eval != use_in_eval:
                    raise RuntimeError(
                        f"Experience attribute {field} redefined with "
                        f"incongruent use_in_eval field. Was "
                        f"{v.use_in_eval}, overridden with {use_in_train}."
                    )
            else:
                setattr(
                    self,
                    field,
                    ExperienceAttribute(
                        value=v, use_in_train=use_in_train, use_in_eval=use_in_eval
                    ),
                )

    def train(self: TCLExperience) -> TCLExperience:
        """Return training experience.

        This is a copy of the experience itself where the private data (e.g.
        experience IDs) is removed to avoid its use during training.
        """
        exp = copy(self)
        exp._exp_mode = ExperienceMode.TRAIN
        return exp

    def eval(self: TCLExperience) -> TCLExperience:
        """Return inference experience.

        This is a copy of the experience itself where the inference data (e.g.
        experience IDs) is available.
        """
        exp = copy(self)
        exp._exp_mode = ExperienceMode.EVAL
        return exp

    def logging(self: TCLExperience) -> TCLExperience:
        """Return logging experience.

        This is a copy of the experience itself where all the attributes are
        available. Useful for logging and metric computations.
        """
        exp = copy(self)
        exp._exp_mode = ExperienceMode.LOGGING
        return exp

    @staticmethod
    def _check_unset_attribute(attribute_name: str, attribute_value: Any):
        assert attribute_value is not None, (
            f"Attribute {attribute_name} "
            + "not set. This is an unexpected and usually liked to errors "
            + "in the implementation of the stream's experience factory."
        )