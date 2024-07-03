def _validate_coefficients(coeffs, n_inputs, can_be_list=True):
    """Helper function to validate input coefficients of plotting functions.

    Args:
        coeffs (array[complex]): A set (or list of sets) of Fourier coefficients of a
            n_inputs-dimensional function.
        n_inputs (int): The number of inputs (dimension) of the function the coefficients are for.
        can_be_list (bool): Whether or not the plotting function accepts a list of
            coefficients, or only a single set.

    Raises:
        TypeError: If the coefficients are not a list or array.
        ValueError: if the coefficients are not a suitable type for the plotting function.
    """
    # Make sure we have a list or numpy array
    if not isinstance(coeffs, list) and not isinstance(coeffs, np.ndarray):
        raise TypeError(
            "Input to coefficient plotting functions must be a list of numerical "
            f"Fourier coefficients. Received input of type {type(coeffs)}"
        )

    # In case we have a list, turn it into a numpy array
    if isinstance(coeffs, list):
        coeffs = np.array(coeffs)

    # Check if the user provided a single set of coefficients to a function that is
    # meant to accept multiple samples; add an extra dimension around it if needed
    if len(coeffs.shape) == n_inputs and can_be_list:
        coeffs = np.array([coeffs])

    # Check now that we have the right number of axes for the type of function
    required_shape_size = n_inputs + 1 if can_be_list else n_inputs
    if len(coeffs.shape) != required_shape_size:
        raise ValueError(
            f"Plotting function expected a list of {n_inputs}-dimensional inputs. "
            f"Received coefficients of {len(coeffs.shape)}-dimensional function."
        )

    # Size of each sample dimension must be 2d_i + 1 where d_i is the i-th degree
    dims = coeffs.shape[1:] if can_be_list else coeffs.shape
    if any((dim - 1) % 2 for dim in dims):
        raise ValueError(
            "Shape of input coefficients must be 2d_i + 1, where d_i is the largest frequency "
            f"in the i-th input. Coefficient array with shape {coeffs.shape} is invalid."
        )

    # Return the coefficients; we may have switched to a numpy array or added a needed extra dimension
    return coeffs