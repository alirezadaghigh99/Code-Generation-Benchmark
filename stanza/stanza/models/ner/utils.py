def is_basic_scheme(all_tags):
    """
    Check if a basic tagging scheme is used. Return True if so.

    Args:
        all_tags: a list of NER tags

    Returns:
        True if the tagging scheme does not use B-, I-, etc, otherwise False
    """
    for tag in all_tags:
        if len(tag) > 2 and tag[:2] in ('B-', 'I-', 'S-', 'E-', 'B_', 'I_', 'S_', 'E_'):
            return False
    return True

