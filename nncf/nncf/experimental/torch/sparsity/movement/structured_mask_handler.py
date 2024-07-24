class StructuredMaskContextGroup:
    """
    Stores together the structured mask contexts that are related to the same building block.
    """

    def __init__(self, group_id: int, structured_mask_contexts: List[StructuredMaskContext]):
        """
        Initializes a group of related structured mask contexts.

        :param group_id: The index of the building block.
        :param structured_mask_contexts: A list of structured mask contexts corresponding
            to the building block.
        """
        self.group_id = group_id
        self.structured_mask_contexts = structured_mask_contexts

    def __str__(self) -> str:
        if not self.structured_mask_contexts:
            ctxes_str = "[]"
        else:
            ctxes = (f"\n\t{ctx}" for ctx in self.structured_mask_contexts)
            ctxes_str = "[{}\n]".format("".join(ctxes))
        return f"{self.__class__.__name__}[{self.group_id}]: {ctxes_str}"

class StructuredMaskContext:
    """
    Context to interact with the operand of a module in movement sparsity.

    This context can resolve the independent structured mask from operand, and can refresh the binary
    mask back to operand with dependent structured mask. Serves as an agent for `StructuredMaskHandler`
    to conduct structured mask resolution.
    """

    def __init__(
        self,
        sparsifier_operand: MovementSparsifier,
        module_node_name: NNCFNodeName,
        grid_size: Tuple[int, int],
        prune_by_row: bool,
    ):
        """
        Initializes the context of the target module for structured masking.

        :param sparsifier_operand: Operand for the target module.
        :param module_node_name: Node name of the target module.
        :param grid_size: The grid shape for resolving the independent structured mask.
        :param prune_by_row: Determines whether to resolve the independent structured mask by row or column.
        """
        self.sparsifier_operand = sparsifier_operand
        self.module_node_name = module_node_name
        operand_mask: torch.Tensor = sparsifier_operand.weight_ctx.binary_mask
        self.operand_mask_shape = operand_mask.shape
        self.grid_size = self._resolve_grid_size(grid_size)
        self.structured_mask_shape = torch.Size(
            dim // grid for dim, grid in zip(self.operand_mask_shape, self.grid_size)
        )
        self.prune_by_row = prune_by_row
        self._independent_structured_mask = None
        self._dependent_structured_mask = None

    def __str__(self) -> str:
        prune_info = "row prune" if self.prune_by_row else "column prune"
        return f'{self.__class__.__name__}({prune_info} by {self.grid_size}, "{self.module_node_name}")'

    @property
    def independent_structured_mask(self) -> Optional[torch.Tensor]:
        if self._independent_structured_mask is None:
            nncf_logger.debug("Independent structured mask has not been calculated. Return None.")
        return self._independent_structured_mask

    @independent_structured_mask.setter
    @torch.no_grad()
    def independent_structured_mask(self, tensor: torch.Tensor):
        if self.structured_mask_shape != tensor.shape:
            raise ValueError("Wrong shape about independent structured mask.")
        if self._independent_structured_mask is None:
            self._independent_structured_mask = tensor.clone()
        else:
            if self._independent_structured_mask.device != tensor.device:
                nncf_logger.debug(f"Changing independent_structured_mask device to {tensor.device}")
                self._independent_structured_mask = self._independent_structured_mask.to(tensor.device)
            self._independent_structured_mask.copy_(tensor)

    @property
    def dependent_structured_mask(self) -> Optional[torch.Tensor]:
        if self._dependent_structured_mask is None:
            nncf_logger.debug("Dependent structured mask has not been calculated. Return None.")
        return self._dependent_structured_mask

    @dependent_structured_mask.setter
    @torch.no_grad()
    def dependent_structured_mask(self, tensor: torch.Tensor):
        if self.structured_mask_shape != tensor.shape:
            raise ValueError("Wrong shape about dependent structured mask.")
        if self._dependent_structured_mask is None:
            self._dependent_structured_mask = tensor.clone()
        else:
            if self._dependent_structured_mask.device != tensor.device:
                nncf_logger.debug(
                    f"Changing dependent_structured_mask device to {tensor.device}",
                )
                self._dependent_structured_mask = self._dependent_structured_mask.to(tensor.device)
            self._dependent_structured_mask.copy_(tensor)

    @torch.no_grad()
    def update_independent_structured_mask_from_operand(self):
        """
        Gets the current unstructured binary mask from operand, resolves it to the independent structured one, and
        stores in `self.independent_structured_mask` for later use in `StructuredMaskHandler`.
        """
        weight_binary_mask = self.sparsifier_operand.weight_ctx.binary_mask.detach().clone()
        mask_by_grid = F.max_pool2d(
            weight_binary_mask.unsqueeze(0), kernel_size=self.grid_size, stride=self.grid_size
        ).squeeze(0)
        preserved_cols = mask_by_grid.amax(dim=0)
        preserved_rows = mask_by_grid.amax(dim=1)

        if self.sparsifier_operand.prune_bias:
            bias_binary_mask = self.sparsifier_operand.bias_ctx.binary_mask.detach().clone()
            bias_preserved_rows = F.max_pool1d(
                bias_binary_mask.view(1, -1), kernel_size=self.grid_size[0], stride=self.grid_size[0]
            ).squeeze(0)
            preserved_rows = bias_preserved_rows.logical_or(preserved_rows)

        structured_mask = preserved_rows.unsqueeze(1) * preserved_cols.unsqueeze(0)
        self.independent_structured_mask = structured_mask
        return structured_mask

    def initialize_binary_mask(self):
        """
        Initialize binary mask by all ones. The inflated dependent mask will be applied via logical "and"
        operation to it. It's needed for the case when 1 binary mask shared for 2 groups: in one group operator
        can be pruned by input channels, i.e. be a consumer of pruning masks, and for another - can be pruned by
        output channels, i.e. be a producer of pruning masks.
        Initial  |  Mask 2 last input channels   |  Mask middle output channel    |     Result
        --------------------------------------------------------------------------------------
         1111                 1100                             1111                     1100
         1111    &            1100               &             0000                =    0000
         1111                 1100                             1111                     1100
        """

        self.sparsifier_operand.weight_ctx.binary_mask.fill_(1)
        if self.sparsifier_operand.prune_bias:
            self.sparsifier_operand.bias_ctx.binary_mask.fill_(1)

    def populate_dependent_structured_mask_to_operand(self):
        """
        Updates the actual binary masks in operand with `self.dependent_structured_mask`.
        """
        structured_mask_inflated = self._inflate_structured_mask(self.dependent_structured_mask, self.grid_size)
        self.sparsifier_operand.weight_ctx.binary_mask *= structured_mask_inflated
        if self.sparsifier_operand.prune_bias:
            self.sparsifier_operand.bias_ctx.binary_mask *= structured_mask_inflated.amax(dim=1)

    def gather_statistics_from_operand(self) -> StructuredMaskContextStatistics:
        """
        Collects the structured mask statistics from the binary masks in operand.

        :return: The statistics of the structured mask context.
        """
        node = self.sparsifier_operand.target_module_node
        assert isinstance(node.layer_attributes, tuple(EXPECTED_NODE_LAYER_ATTRS))
        weight_shape: Tuple[int, int] = tuple(node.layer_attributes.get_weight_shape())
        bias_shape: Tuple[int] = (
            (node.layer_attributes.get_bias_shape(),) if self.sparsifier_operand.prune_bias else (0,)
        )

        pruned_weight_shape = list(weight_shape)
        head_id_to_keep = []
        if self.prune_by_row:
            prunable_rows = self.sparsifier_operand.weight_ctx.binary_mask.amax(dim=1)
            pruned_weight_shape[0] = int(prunable_rows.count_nonzero().item())
            kept_row_blocks = F.max_pool1d(prunable_rows.unsqueeze(0), kernel_size=self.grid_size[0]).squeeze(0)
            head_id_to_keep = kept_row_blocks.nonzero().view(-1).cpu().numpy().tolist()
        else:
            prunable_cols = self.sparsifier_operand.weight_ctx.binary_mask.amax(dim=0)
            pruned_weight_shape[1] = int(prunable_cols.count_nonzero().item())
            kept_col_blocks = F.max_pool1d(prunable_cols.unsqueeze(0), kernel_size=self.grid_size[1]).squeeze(0)
            head_id_to_keep = kept_col_blocks.nonzero().view(-1).cpu().numpy().tolist()

        pruned_bias_shape = bias_shape
        if self.sparsifier_operand.prune_bias and self.prune_by_row:
            pruned_bias_shape = (int(self.sparsifier_operand.bias_ctx.binary_mask.count_nonzero().item()),)

        return StructuredMaskContextStatistics(
            weight_shape=weight_shape,
            pruned_weight_shape=tuple(pruned_weight_shape),
            bias_shape=bias_shape,
            pruned_bias_shape=pruned_bias_shape,
            head_or_channel_id_to_keep=head_id_to_keep,
            module_node_name=self.module_node_name,
        )

    def _resolve_grid_size(self, grid_size) -> Tuple[int, int]:
        a, b = grid_size
        return (a if a > 0 else self.operand_mask_shape[0], b if b > 0 else self.operand_mask_shape[1])

    @staticmethod
    def _inflate_structured_mask(structured_mask: torch.Tensor, grid_size: Tuple[int, int]) -> torch.Tensor:
        assert len(structured_mask.shape) == len(
            grid_size
        ), f"Unmatched dimension with structured_mask in shape {structured_mask.shape} and grid_size in 2D."
        inflated_mask = structured_mask.clone()
        for axis, repeat_times in enumerate(grid_size):
            inflated_mask = inflated_mask.repeat_interleave(repeat_times, dim=axis)
        return inflated_mask

