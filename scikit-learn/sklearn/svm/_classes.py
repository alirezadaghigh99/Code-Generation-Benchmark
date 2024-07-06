def _validate_dual_parameter(dual, loss, penalty, multi_class, X):
    """Helper function to assign the value of dual parameter."""
    if dual == "auto":
        if X.shape[0] < X.shape[1]:
            try:
                _get_liblinear_solver_type(multi_class, penalty, loss, True)
                return True
            except ValueError:  # dual not supported for the combination
                return False
        else:
            try:
                _get_liblinear_solver_type(multi_class, penalty, loss, False)
                return False
            except ValueError:  # primal not supported by the combination
                return True
    else:
        return dual

