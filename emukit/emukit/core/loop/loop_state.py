def create_loop_state(x_init: np.ndarray, y_init: np.ndarray, **kwargs) -> LoopState:
    """
    Creates a loop state object using the provided data

    :param x_init: x values for initial function evaluations. Shape: (n_initial_points x n_input_dims)
    :param y_init: y values for initial function evaluations. Shape: (n_initial_points x n_output_dims)
    :param kwargs: extra outputs observed from a function evaluation. Shape: (n_initial_points x n_dims)
    """
    if x_init.shape[0] != y_init.shape[0]:
        error_message = "X and Y should have the same length. Actual length x_init {}, y_init {}".format(
            x_init.shape[0], y_init.shape[0]
        )
        raise ValueError(error_message)

    for key, value in kwargs.items():
        if value.shape[0] != x_init.shape[0]:
            raise ValueError(
                "Expected keyword argument {} to have length {} but actual length is {}".format(
                    key, x_init.shape[0], value.shape[0]
                )
            )

    initial_results = []
    for i in range(x_init.shape[0]):
        kwargs_dict = dict([(key, vals[i]) for key, vals in kwargs.items()])
        initial_results.append(UserFunctionResult(x_init[i], y_init[i], **kwargs_dict))

    return LoopState(initial_results)

