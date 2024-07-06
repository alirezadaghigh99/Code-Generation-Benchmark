def get_random_string(n: int = 5):
    """Return random string at the given size.

    Parameters
    ----------
    n : int , default: 5
        the size of the string to return.

    Returns
    -------
    str
        a random string
    """
    return ''.join(random.choices(ascii_uppercase + digits, k=n))

