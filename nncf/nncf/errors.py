class ValidationError(Exception):
    """
    Raised when an internal NNCF validation check fails,
    for example, if the user supplied an invalid or inconsistent set of arguments.
    """

    pass

class InternalError(Exception):
    """
    Raised when an internal error occurs within the NNCF framework.

    This exception is raised when an unexpected internal error occurs during the execution
    of NNCF. It indicates a situation where the code encountered an unexpected condition.

    """

    pass

