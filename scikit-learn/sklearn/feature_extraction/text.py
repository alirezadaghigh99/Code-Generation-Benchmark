def strip_accents_unicode(s):
    """Transform accentuated unicode symbols into their simple counterpart.

    Warning: the python-level loop and join operations make this
    implementation 20 times slower than the strip_accents_ascii basic
    normalization.

    Parameters
    ----------
    s : str
        The string to strip.

    Returns
    -------
    s : str
        The stripped string.

    See Also
    --------
    strip_accents_ascii : Remove accentuated char for any unicode symbol that
        has a direct ASCII equivalent.
    """
    try:
        # If `s` is ASCII-compatible, then it does not contain any accented
        # characters and we can avoid an expensive list comprehension
        s.encode("ASCII", errors="strict")
        return s
    except UnicodeEncodeError:
        normalized = unicodedata.normalize("NFKD", s)
        return "".join([c for c in normalized if not unicodedata.combining(c)])

