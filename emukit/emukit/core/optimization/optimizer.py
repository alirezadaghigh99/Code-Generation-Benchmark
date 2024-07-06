def apply_optimizer(
    optimizer: Optimizer,
    x0: np.ndarray,
    space: ParameterSpace,
    f: Callable = None,
    df: Callable = None,
    f_df: Callable = None,
    context_manager: ContextManager = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimizes f using the optimizer supplied, deals with potential context variables.

    :param optimizer: The optimizer object that will perform the optimization
    :param x0: initial point for a local optimizer (x0 can be defined with or without the context included).
    :param f: function to optimize.
    :param df: gradient of the function to optimize.
    :param f_df: returns both the function to optimize and its gradient.
    :param context_manager: If provided, x0 (and the optimizer) operates in the space without the context
    :param space: Parameter space describing input domain, including any context variables
    :return: Location of optimum and value at optimum
    """

    if context_manager is None:
        context_manager = ContextManager(space, {})

    # Compute new objective that inputs non context variables but takes into account the values of the context ones.
    # It does nothing if no context is passed
    problem = OptimizationWithContext(x0=x0, f=f, df=df, f_df=f_df, context_manager=context_manager)

    add_context = lambda x: context_manager.expand_vector(x)

    # Optimize point
    if f is None:
        f_no_context = None
    else:
        f_no_context = problem.f_no_context

    if df is None:
        df_no_context = None
    else:
        df_no_context = problem.df_no_context

    if f_df is None:
        f_df_no_context = None
    else:
        f_df_no_context = problem.f_df_no_context

    optimized_x, _ = optimizer.optimize(problem.x0_no_context, f_no_context, df_no_context, f_df_no_context)

    # Add context and round according to the type of variables of the design space
    suggested_x_with_context = add_context(optimized_x)
    suggested_x_with_context_rounded = space.round(suggested_x_with_context)

    if f is None:
        f_opt, _ = f_df(suggested_x_with_context_rounded)
    else:
        f_opt = f(suggested_x_with_context_rounded)
    return suggested_x_with_context_rounded, f_opt

