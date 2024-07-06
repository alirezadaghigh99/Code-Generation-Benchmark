def Kuu(inducing_variable, kernel, jitter=None):
    func = gpflow.covariances.Kuu.dispatch(
        gpflow.inducing_variables.InducingPoints, gpflow.kernels.Kernel
    )
    return func(inducing_variable, kernel.base_kernel, jitter=jitter)

def Kuf(inducing_variable, kernel, a_input):
    return kernel.base_kernel(inducing_variable.Z, kernel.cnn(a_input))

