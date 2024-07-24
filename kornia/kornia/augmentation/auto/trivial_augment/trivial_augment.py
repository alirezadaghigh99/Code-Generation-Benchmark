class TrivialAugment(PolicyAugmentBase):
    """Apply TrivialAugment :cite:`muller2021trivialaugment` augmentation strategies.

    Args:
        policy: candidate transformations. If None, a default candidate list will be used.
        transformation_matrix_mode: computation mode for the chained transformation matrix, via `.transform_matrix`
                                    attribute.
                                    If `silent`, transformation matrix will be computed silently and the non-rigid
                                    modules will be ignored as identity transformations.
                                    If `rigid`, transformation matrix will be computed silently and the non-rigid
                                    modules will trigger errors.
                                    If `skip`, transformation matrix will be totally ignored.

    Examples:
        >>> import kornia.augmentation as K
        >>> in_tensor = torch.rand(5, 3, 30, 30)
        >>> aug = K.AugmentationSequential(TrivialAugment())
        >>> aug(in_tensor).shape
        torch.Size([5, 3, 30, 30])
    """

    def __init__(
        self, policy: Optional[List[SUBPOLICY_CONFIG]] = None, transformation_matrix_mode: str = "silent"
    ) -> None:
        if policy is None:
            _policy = default_policy
        else:
            _policy = policy

        super().__init__(_policy, transformation_matrix_mode=transformation_matrix_mode)
        selection_weights = torch.tensor([1.0 / len(self)] * len(self))
        self.rand_selector = Categorical(selection_weights)

    def compose_subpolicy_sequential(self, subpolicy: SUBPOLICY_CONFIG) -> PolicySequential:
        if len(subpolicy) != 1:
            raise RuntimeError(f"Each policy must have only one operation for TrivialAugment. Got {len(subpolicy)}.")
        name, low, high = subpolicy[0]
        return PolicySequential(*[getattr(ops, name)(low, high)])

    def get_forward_sequence(self, params: Optional[List[ParamItem]] = None) -> Iterator[Tuple[str, Module]]:
        if params is None:
            idx = self.rand_selector.sample((1,))
            return self.get_children_by_indices(idx)

        return self.get_children_by_params(params)

