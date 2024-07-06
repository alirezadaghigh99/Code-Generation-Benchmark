def tensordot(  # noqa: F811
    a,
    b,
    dims=2,
    out: Optional[torch.Tensor] = None,
):
    r"""Returns a contraction of a and b over multiple dimensions.

    :attr:`tensordot` implements a generalized matrix product.

    Args:
      a (Tensor): Left tensor to contract
      b (Tensor): Right tensor to contract
      dims (int or Tuple[List[int], List[int]] or List[List[int]] containing two lists or Tensor): number of dimensions to
         contract or explicit lists of dimensions for :attr:`a` and
         :attr:`b` respectively

    When called with a non-negative integer argument :attr:`dims` = :math:`d`, and
    the number of dimensions of :attr:`a` and :attr:`b` is :math:`m` and :math:`n`,
    respectively, :func:`~torch.tensordot` computes

    .. math::
        r_{i_0,...,i_{m-d}, i_d,...,i_n}
          = \sum_{k_0,...,k_{d-1}} a_{i_0,...,i_{m-d},k_0,...,k_{d-1}} \times b_{k_0,...,k_{d-1}, i_d,...,i_n}.

    When called with :attr:`dims` of the list form, the given dimensions will be contracted
    in place of the last :math:`d` of :attr:`a` and the first :math:`d` of :math:`b`. The sizes
    in these dimensions must match, but :func:`~torch.tensordot` will deal with broadcasted
    dimensions.

    Examples::

        >>> a = torch.arange(60.).reshape(3, 4, 5)
        >>> b = torch.arange(24.).reshape(4, 3, 2)
        >>> torch.tensordot(a, b, dims=([1, 0], [0, 1]))
        tensor([[4400., 4730.],
                [4532., 4874.],
                [4664., 5018.],
                [4796., 5162.],
                [4928., 5306.]])

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> a = torch.randn(3, 4, 5, device='cuda')
        >>> b = torch.randn(4, 5, 6, device='cuda')
        >>> c = torch.tensordot(a, b, dims=2).cpu()
        tensor([[ 8.3504, -2.5436,  6.2922,  2.7556, -1.0732,  3.2741],
                [ 3.3161,  0.0704,  5.0187, -0.4079, -4.3126,  4.8744],
                [ 0.8223,  3.9445,  3.2168, -0.2400,  3.4117,  1.7780]])

        >>> a = torch.randn(3, 5, 4, 6)
        >>> b = torch.randn(6, 4, 5, 3)
        >>> torch.tensordot(a, b, dims=([2, 1, 3], [1, 2, 0]))
        tensor([[  7.7193,  -2.4867, -10.3204],
                [  1.5513, -14.4737,  -6.5113],
                [ -0.2850,   4.2573,  -3.5997]])
    """
    if has_torch_function_variadic(a, b):
        return handle_torch_function(tensordot, (a, b), a, b, dims=dims, out=out)

    if not isinstance(dims, (tuple, list, torch.Tensor, int, torch.SymInt)):
        raise RuntimeError(
            "tensordot expects dims to be int or "
            + "Tuple[List[int], List[int]] or "
            + "List[List[int]] containing two lists, but got "
            + f"dims={dims}"
        )

    dims_a: List[int] = []
    dims_b: List[int] = []

    if isinstance(dims, (tuple, list)):
        dims_a, dims_b = dims

    if isinstance(dims, torch.Tensor):
        num_elements = dims.numel()
        if num_elements > 1:
            assert dims.size()[0] == 2
            dims_a = torch.jit.annotate(List[int], dims[0].tolist())
            dims_b = torch.jit.annotate(List[int], dims[1].tolist())
        else:
            dims_val = int(dims.item())
            if dims_val < 0:
                raise RuntimeError(f"tensordot expects dims >= 0, but got dims={dims}")
            dims_a = list(range(-dims_val, 0))
            dims_b = list(range(dims_val))

    if isinstance(dims, (int, torch.SymInt)):
        if dims < 0:
            raise RuntimeError(f"tensordot expects dims >= 0, but got dims={dims}")
        if dims > min(a.dim(), b.dim()):
            raise RuntimeError(
                f"tensordot expects dims < ndim_a or ndim_b, but got dims={dims}"
            )
        dims_a = list(range(-dims, 0))
        dims_b = list(range(dims))

    if out is None:
        return _VF.tensordot(a, b, dims_a, dims_b)  # type: ignore[attr-defined]
    else:
        return _VF.tensordot(a, b, dims_a, dims_b, out=out)  # type: ignore[attr-defined]

def einsum(*args: Any) -> Tensor:
    r"""einsum(equation, *operands) -> Tensor

    Sums the product of the elements of the input :attr:`operands` along dimensions specified using a notation
    based on the Einstein summation convention.

    Einsum allows computing many common multi-dimensional linear algebraic array operations by representing them
    in a short-hand format based on the Einstein summation convention, given by :attr:`equation`. The details of
    this format are described below, but the general idea is to label every dimension of the input :attr:`operands`
    with some subscript and define which subscripts are part of the output. The output is then computed by summing
    the product of the elements of the :attr:`operands` along the dimensions whose subscripts are not part of the
    output. For example, matrix multiplication can be computed using einsum as `torch.einsum("ij,jk->ik", A, B)`.
    Here, j is the summation subscript and i and k the output subscripts (see section below for more details on why).

    Equation:

        The :attr:`equation` string specifies the subscripts (letters in `[a-zA-Z]`) for each dimension of
        the input :attr:`operands` in the same order as the dimensions, separating subscripts for each operand by a
        comma (','), e.g. `'ij,jk'` specify subscripts for two 2D operands. The dimensions labeled with the same subscript
        must be broadcastable, that is, their size must either match or be `1`. The exception is if a subscript is
        repeated for the same input operand, in which case the dimensions labeled with this subscript for this operand
        must match in size and the operand will be replaced by its diagonal along these dimensions. The subscripts that
        appear exactly once in the :attr:`equation` will be part of the output, sorted in increasing alphabetical order.
        The output is computed by multiplying the input :attr:`operands` element-wise, with their dimensions aligned based
        on the subscripts, and then summing out the dimensions whose subscripts are not part of the output.

        Optionally, the output subscripts can be explicitly defined by adding an arrow ('->') at the end of the equation
        followed by the subscripts for the output. For instance, the following equation computes the transpose of a
        matrix multiplication: 'ij,jk->ki'. The output subscripts must appear at least once for some input operand and
        at most once for the output.

        Ellipsis ('...') can be used in place of subscripts to broadcast the dimensions covered by the ellipsis.
        Each input operand may contain at most one ellipsis which will cover the dimensions not covered by subscripts,
        e.g. for an input operand with 5 dimensions, the ellipsis in the equation `'ab...c'` cover the third and fourth
        dimensions. The ellipsis does not need to cover the same number of dimensions across the :attr:`operands` but the
        'shape' of the ellipsis (the size of the dimensions covered by them) must broadcast together. If the output is not
        explicitly defined with the arrow ('->') notation, the ellipsis will come first in the output (left-most dimensions),
        before the subscript labels that appear exactly once for the input operands. e.g. the following equation implements
        batch matrix multiplication `'...ij,...jk'`.

        A few final notes: the equation may contain whitespaces between the different elements (subscripts, ellipsis,
        arrow and comma) but something like `'. . .'` is not valid. An empty string `''` is valid for scalar operands.

    .. note::

        ``torch.einsum`` handles ellipsis ('...') differently from NumPy in that it allows dimensions
        covered by the ellipsis to be summed over, that is, ellipsis are not required to be part of the output.

    .. note::

        This function uses opt_einsum (https://optimized-einsum.readthedocs.io/en/stable/) to speed up computation or to
        consume less memory by optimizing contraction order. This optimization occurs when there are at least three
        inputs, since the order does not matter otherwise. Note that finding _the_ optimal path is an NP-hard problem,
        thus, opt_einsum relies on different heuristics to achieve near-optimal results. If opt_einsum is not available,
        the default order is to contract from left to right.

        To bypass this default behavior, add the following line to disable the usage of opt_einsum and skip path
        calculation: `torch.backends.opt_einsum.enabled = False`

        To specify which strategy you'd like for opt_einsum to compute the contraction path, add the following line:
        `torch.backends.opt_einsum.strategy = 'auto'`. The default strategy is 'auto', and we also support 'greedy' and
        'optimal'. Disclaimer that the runtime of 'optimal' is factorial in the number of inputs! See more details in
        the opt_einsum documentation (https://optimized-einsum.readthedocs.io/en/stable/path_finding.html).

    .. note::

        As of PyTorch 1.10 :func:`torch.einsum` also supports the sublist format (see examples below). In this format,
        subscripts for each operand are specified by sublists, list of integers in the range [0, 52). These sublists
        follow their operands, and an extra sublist can appear at the end of the input to specify the output's
        subscripts., e.g. `torch.einsum(op1, sublist1, op2, sublist2, ..., [subslist_out])`. Python's `Ellipsis` object
        may be provided in a sublist to enable broadcasting as described in the Equation section above.

    Args:
        equation (str): The subscripts for the Einstein summation.
        operands (List[Tensor]): The tensors to compute the Einstein summation of.

    Examples::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> # trace
        >>> torch.einsum('ii', torch.randn(4, 4))
        tensor(-1.2104)

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> # diagonal
        >>> torch.einsum('ii->i', torch.randn(4, 4))
        tensor([-0.1034,  0.7952, -0.2433,  0.4545])

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> # outer product
        >>> x = torch.randn(5)
        >>> y = torch.randn(4)
        >>> torch.einsum('i,j->ij', x, y)
        tensor([[ 0.1156, -0.2897, -0.3918,  0.4963],
                [-0.3744,  0.9381,  1.2685, -1.6070],
                [ 0.7208, -1.8058, -2.4419,  3.0936],
                [ 0.1713, -0.4291, -0.5802,  0.7350],
                [ 0.5704, -1.4290, -1.9323,  2.4480]])

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> # batch matrix multiplication
        >>> As = torch.randn(3, 2, 5)
        >>> Bs = torch.randn(3, 5, 4)
        >>> torch.einsum('bij,bjk->bik', As, Bs)
        tensor([[[-1.0564, -1.5904,  3.2023,  3.1271],
                [-1.6706, -0.8097, -0.8025, -2.1183]],

                [[ 4.2239,  0.3107, -0.5756, -0.2354],
                [-1.4558, -0.3460,  1.5087, -0.8530]],

                [[ 2.8153,  1.8787, -4.3839, -1.2112],
                [ 0.3728, -2.1131,  0.0921,  0.8305]]])

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> # with sublist format and ellipsis
        >>> torch.einsum(As, [..., 0, 1], Bs, [..., 1, 2], [..., 0, 2])
        tensor([[[-1.0564, -1.5904,  3.2023,  3.1271],
                [-1.6706, -0.8097, -0.8025, -2.1183]],

                [[ 4.2239,  0.3107, -0.5756, -0.2354],
                [-1.4558, -0.3460,  1.5087, -0.8530]],

                [[ 2.8153,  1.8787, -4.3839, -1.2112],
                [ 0.3728, -2.1131,  0.0921,  0.8305]]])

        >>> # batch permute
        >>> A = torch.randn(2, 3, 4, 5)
        >>> torch.einsum('...ij->...ji', A).shape
        torch.Size([2, 3, 5, 4])

        >>> # equivalent to torch.nn.functional.bilinear
        >>> A = torch.randn(3, 5, 4)
        >>> l = torch.randn(2, 5)
        >>> r = torch.randn(2, 4)
        >>> torch.einsum('bn,anm,bm->ba', l, A, r)
        tensor([[-0.3430, -5.2405,  0.4494],
                [ 0.3311,  5.5201, -3.0356]])
    """
    import torch.backends.opt_einsum as opt_einsum

    # This wrapper exists to support variadic args.
    if len(args) < 2:
        raise ValueError(
            "einsum(): must specify the equation string and at least one operand, "
            "or at least one operand and its subscripts list"
        )

    equation = None
    operands = None

    if isinstance(args[0], torch.Tensor):
        # Convert the subscript list format which is an interleaving of operand and its subscripts
        # list with an optional output subscripts list at the end (see documentation for more details on this)
        # to the equation string format by creating the equation string from the subscripts list and grouping the
        # input operands into a tensorlist (List[Tensor]).
        def parse_subscript(n: int) -> str:
            if n == Ellipsis:
                return "..."
            if n >= 0 and n < 26:
                return chr(ord("A") + n)
            if n >= 26 and n < 52:
                return chr(ord("a") + n - 26)
            raise ValueError(
                "einsum(): subscript in subscript list is not within the valid range [0, 52)"
            )

        # Parse subscripts for input operands
        equation = ",".join("".join(parse_subscript(s) for s in l) for l in args[1::2])

        # Parse optional output subscripts (provided when the number of arguments is odd)
        if len(args) % 2 == 1:
            equation += "->" + "".join(parse_subscript(s) for s in args[-1])
            operands = args[:-1:2]
        else:
            operands = args[::2]
    else:
        equation = args[0]
        operands = args[1:]

    if has_torch_function(operands):
        return handle_torch_function(einsum, operands, equation, *operands)

    if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
        # the old interface of passing the operands as one list argument
        _operands = operands[0]
        # recurse incase operands contains value that has torch function
        # in the original implementation this line is omitted
        return einsum(equation, *_operands)

    if len(operands) <= 2 or not opt_einsum.enabled:
        # the path for contracting 0 or 1 time(s) is already optimized
        # or the user has disabled using opt_einsum
        return _VF.einsum(equation, operands)  # type: ignore[attr-defined]

    path = None
    if opt_einsum.is_available():
        _opt_einsum = opt_einsum.get_opt_einsum()
        tupled_path = _opt_einsum.contract_path(
            equation, *operands, optimize=opt_einsum.strategy
        )[0]
        # flatten path for dispatching to C++
        path = [item for pair in tupled_path for item in pair]
    return _VF.einsum(equation, operands, path=path)  # type: ignore[attr-defined]

def meshgrid(
        *tensors: Union[Tensor, List[Tensor]], indexing: Optional[str] = None
    ) -> Tuple[Tensor, ...]:
        return _meshgrid(*tensors, indexing=indexing)

