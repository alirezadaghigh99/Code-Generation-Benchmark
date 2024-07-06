def make_non_linear_kernels(
    base_kernel_class: Type[GPy.kern.Kern], n_fidelities: int, n_input_dims: int, ARD: bool = False
) -> List:
    """
    This function takes a base kernel class and constructs the structured multi-fidelity kernels

    At the first level the kernel is simply:
    .. math
        k_{base}(x, x')

    At subsequent levels the kernels are of the form
    .. math
        k_{base}(x, x')k_{base}(y_{i-1}, y{i-1}') + k_{base}(x, x')

    :param base_kernel_class: GPy class definition of the kernel type to construct the kernels at
    :param n_fidelities: Number of fidelities in the model. A kernel will be returned for each fidelity
    :param n_input_dims: The dimensionality of the input.
    :param ARD: If True, uses different lengthscales for different dimensions. Otherwise the same lengthscale is used
                for all dimensions. Default False.
    :return: A list of kernels with one entry for each fidelity starting from lowest to highest fidelity.
    """

    base_dims_list = list(range(n_input_dims))
    kernels = [base_kernel_class(n_input_dims, active_dims=base_dims_list, ARD=ARD, name="kern_fidelity_1")]
    for i in range(1, n_fidelities):
        fidelity_name = "fidelity" + str(i + 1)
        interaction_kernel = base_kernel_class(
            n_input_dims, active_dims=base_dims_list, ARD=ARD, name="scale_kernel_" + fidelity_name
        )
        scale_kernel = base_kernel_class(1, active_dims=[n_input_dims], name="previous_fidelity_" + fidelity_name)
        bias_kernel = base_kernel_class(
            n_input_dims, active_dims=base_dims_list, ARD=ARD, name="bias_kernel_" + fidelity_name
        )
        kernels.append(interaction_kernel * scale_kernel + bias_kernel)
    return kernels

