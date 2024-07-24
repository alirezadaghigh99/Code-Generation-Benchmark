class TensorNonTensorSeparator(object):
    """
    Class that provides function to separate/combine tensors and nontensors
    parameters.

    Examples
    --------
    >>> import torch
    >>> from deepchem.utils.pytorch_utils import TensorNonTensorSeparator
    >>> a = torch.tensor([1.,2,3])
    >>> b = 4.
    >>> c = torch.tensor([5.,6,7], requires_grad=True)
    >>> params = [a, b, c]
    >>> separator = TensorNonTensorSeparator(params)
    >>> tensor_params = separator.get_tensor_params()
    >>> tensor_params
    [tensor([5., 6., 7.], requires_grad=True)]

    """

    def __init__(self, params: Sequence, varonly: bool = True):
        """Initialize the TensorNonTensorSeparator.

        Parameters
        ----------
        params: Sequence
            A list of tensor or non-tensor parameters.
        varonly: bool
            If True, only tensor parameters with requires_grad=True will be
            returned. Otherwise, all tensor parameters will be returned.

        """
        self.tensor_idxs = []
        self.tensor_params = []
        self.nontensor_idxs = []
        self.nontensor_params = []
        self.nparams = len(params)
        for (i, p) in enumerate(params):
            if isinstance(p, torch.Tensor) and ((varonly and p.requires_grad) or
                                                (not varonly)):
                self.tensor_idxs.append(i)
                self.tensor_params.append(p)
            else:
                self.nontensor_idxs.append(i)
                self.nontensor_params.append(p)
        self.alltensors = len(self.tensor_idxs) == self.nparams

    def get_tensor_params(self):
        """Returns a list of tensor parameters.

        Returns
        -------
        List[torch.Tensor]
            A list of tensor parameters.

        """
        return self.tensor_params

    def ntensors(self):
        """Returns the number of tensor parameters.

        Returns
        -------
        int
            The number of tensor parameters.

        """
        return len(self.tensor_idxs)

    def nnontensors(self):
        """Returns the number of nontensor parameters.

        Returns
        -------
        int
            The number of nontensor parameters.

        """
        return len(self.nontensor_idxs)

    def reconstruct_params(self, tensor_params, nontensor_params=None):
        """Reconstruct the parameters from tensor and nontensor parameters.

        Parameters
        ----------
        tensor_params: List[torch.Tensor]
            A list of tensor parameters.
        nontensor_params: Optional[List]
            A list of nontensor parameters. If None, the original nontensor
            parameters will be used.

        Returns
        -------
        List
            A list of parameters.

        """
        if nontensor_params is None:
            nontensor_params = self.nontensor_params
        if len(tensor_params) + len(nontensor_params) != self.nparams:
            raise ValueError(
                "The total length of tensor and nontensor params "
                "do not match with the expected length: %d instead of %d" %
                (len(tensor_params) + len(nontensor_params), self.nparams))
        if self.alltensors:
            return tensor_params

        params = [None for _ in range(self.nparams)]
        for nidx, p in zip(self.nontensor_idxs, nontensor_params):
            params[nidx] = p
        for idx, p in zip(self.tensor_idxs, tensor_params):
            params[idx] = p
        return params

