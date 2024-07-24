class L1Unstructured(BasePruningMethod):
    r"""Prune (currently unpruned) units in a tensor by zeroing out the ones with the lowest L1-norm.

    Args:
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
    """

    PRUNING_TYPE = "unstructured"

    def __init__(self, amount):
        # Check range of validity of pruning amount
        _validate_pruning_amount_init(amount)
        self.amount = amount

    def compute_mask(self, t, default_mask):
        # Check that the amount of units to prune is not > than the number of
        # parameters in t
        tensor_size = t.nelement()
        # Compute number of units to prune: amount if int,
        # else amount * tensor_size
        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        _validate_pruning_amount(nparams_toprune, tensor_size)

        mask = default_mask.clone(memory_format=torch.contiguous_format)

        if nparams_toprune != 0:  # k=0 not supported by torch.kthvalue
            # largest=True --> top k; largest=False --> bottom k
            # Prune the smallest k
            topk = torch.topk(torch.abs(t).view(-1), k=nparams_toprune, largest=False)
            # topk will have .indices and .values
            mask.view(-1)[topk.indices] = 0

        return mask

    @classmethod
    def apply(cls, module, name, amount, importance_scores=None):
        r"""Add pruning on the fly and reparametrization of a tensor.

        Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the
                absolute number of parameters to prune.
            importance_scores (torch.Tensor): tensor of importance scores (of same
                shape as module parameter) used to compute mask for pruning.
                The values in this tensor indicate the importance of the corresponding
                elements in the parameter being pruned.
                If unspecified or None, the module parameter will be used in its place.
        """
        return super().apply(
            module, name, amount=amount, importance_scores=importance_scores
        )

class PruningContainer(BasePruningMethod):
    """Container holding a sequence of pruning methods for iterative pruning.

    Keeps track of the order in which pruning methods are applied and handles
    combining successive pruning calls.

    Accepts as argument an instance of a BasePruningMethod or an iterable of
    them.
    """

    def __init__(self, *args):
        self._pruning_methods: Tuple[BasePruningMethod, ...] = tuple()
        if not isinstance(args, Iterable):  # only 1 item
            self._tensor_name = args._tensor_name
            self.add_pruning_method(args)
        elif len(args) == 1:  # only 1 item in a tuple
            self._tensor_name = args[0]._tensor_name
            self.add_pruning_method(args[0])
        else:  # manual construction from list or other iterable (or no args)
            for method in args:
                self.add_pruning_method(method)

    def add_pruning_method(self, method):
        r"""Add a child pruning ``method`` to the container.

        Args:
            method (subclass of BasePruningMethod): child pruning method
                to be added to the container.
        """
        # check that we're adding a pruning method to the container
        if not isinstance(method, BasePruningMethod) and method is not None:
            raise TypeError(f"{type(method)} is not a BasePruningMethod subclass")
        elif method is not None and self._tensor_name != method._tensor_name:
            raise ValueError(
                "Can only add pruning methods acting on "
                f"the parameter named '{self._tensor_name}' to PruningContainer {self}."
                + f" Found '{method._tensor_name}'"
            )
        # if all checks passed, add to _pruning_methods tuple
        self._pruning_methods += (method,)  # type: ignore[operator]

    def __len__(self):
        return len(self._pruning_methods)

    def __iter__(self):
        return iter(self._pruning_methods)

    def __getitem__(self, idx):
        return self._pruning_methods[idx]

    def compute_mask(self, t, default_mask):
        r"""Apply the latest ``method`` by computing the new partial masks and returning its combination with the ``default_mask``.

        The new partial mask should be computed on the entries or channels
        that were not zeroed out by the ``default_mask``.
        Which portions of the tensor ``t`` the new mask will be calculated from
        depends on the ``PRUNING_TYPE`` (handled by the type handler):

        * for 'unstructured', the mask will be computed from the raveled
          list of nonmasked entries;

        * for 'structured', the mask will be computed from the nonmasked
          channels in the tensor;

        * for 'global', the mask will be computed across all entries.

        Args:
            t (torch.Tensor): tensor representing the parameter to prune
                (of same dimensions as ``default_mask``).
            default_mask (torch.Tensor): mask from previous pruning iteration.

        Returns:
            mask (torch.Tensor): new mask that combines the effects
            of the ``default_mask`` and the new mask from the current
            pruning ``method`` (of same dimensions as ``default_mask`` and
            ``t``).
        """

        def _combine_masks(method, t, mask):
            r"""Combine the masks from all pruning methods and returns a new mask.

            Args:
                method (a BasePruningMethod subclass): pruning method
                    currently being applied.
                t (torch.Tensor): tensor representing the parameter to prune
                    (of same dimensions as mask).
                mask (torch.Tensor): mask from previous pruning iteration

            Returns:
                new_mask (torch.Tensor): new mask that combines the effects
                    of the old mask and the new mask from the current
                    pruning method (of same dimensions as mask and t).
            """
            new_mask = mask  # start off from existing mask
            new_mask = new_mask.to(dtype=t.dtype)

            # compute a slice of t onto which the new pruning method will operate
            if method.PRUNING_TYPE == "unstructured":
                # prune entries of t where the mask is 1
                slc = mask == 1

            # for struct pruning, exclude channels that have already been
            # entirely pruned
            elif method.PRUNING_TYPE == "structured":
                if not hasattr(method, "dim"):
                    raise AttributeError(
                        "Pruning methods of PRUNING_TYPE "
                        '"structured" need to have the attribute `dim` defined.'
                    )

                # find the channels to keep by removing the ones that have been
                # zeroed out already (i.e. where sum(entries) == 0)
                n_dims = t.dim()  # "is this a 2D tensor? 3D? ..."
                dim = method.dim
                # convert negative indexing
                if dim < 0:
                    dim = n_dims + dim
                # if dim is still negative after subtracting it from n_dims
                if dim < 0:
                    raise IndexError(
                        f"Index is out of bounds for tensor with dimensions {n_dims}"
                    )
                # find channels along dim = dim that aren't already tots 0ed out
                keep_channel = mask.sum(dim=[d for d in range(n_dims) if d != dim]) != 0
                # create slice to identify what to prune
                slc = [slice(None)] * n_dims
                slc[dim] = keep_channel

            elif method.PRUNING_TYPE == "global":
                n_dims = len(t.shape)  # "is this a 2D tensor? 3D? ..."
                slc = [slice(None)] * n_dims

            else:
                raise ValueError(f"Unrecognized PRUNING_TYPE {method.PRUNING_TYPE}")

            # compute the new mask on the unpruned slice of the tensor t
            partial_mask = method.compute_mask(t[slc], default_mask=mask[slc])
            new_mask[slc] = partial_mask.to(dtype=new_mask.dtype)

            return new_mask

        method = self._pruning_methods[-1]
        mask = _combine_masks(method, t, default_mask)
        return mask

