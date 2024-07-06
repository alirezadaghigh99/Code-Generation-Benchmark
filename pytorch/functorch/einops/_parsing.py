def parse_pattern(
    pattern: str, axes_lengths: Mapping[str, int]
) -> Tuple[ParsedExpression, ParsedExpression]:
    """Parse an `einops`-style pattern into a left-hand side and right-hand side `ParsedExpression` object.

    Args:
        pattern (str): the `einops`-style rearrangement pattern
        axes_lengths (Mapping[str, int]): any additional length specifications for dimensions

    Returns:
       Tuple[ParsedExpression, ParsedExpression]: a tuple containing the left-hand side and right-hand side expressions
    """
    # adapted from einops.einops._prepare_transformation_recipe
    # https://github.com/arogozhnikov/einops/blob/230ac1526c1f42c9e1f7373912c7f8047496df11/einops/einops.py
    try:
        left_str, right_str = pattern.split("->")
    except ValueError:
        raise ValueError("Pattern must contain a single '->' separator") from None

    if _ellipsis in axes_lengths:
        raise ValueError(f"'{_ellipsis}' is not an allowed axis identifier")

    left = ParsedExpression(left_str)
    right = ParsedExpression(right_str)

    if not left.has_ellipsis and right.has_ellipsis:
        raise ValueError(
            f"Ellipsis found in right side, but not left side of a pattern {pattern}"
        )
    if left.has_ellipsis and left.has_ellipsis_parenthesized:
        raise ValueError(
            f"Ellipsis is parenthesis in the left side is not allowed: {pattern}"
        )

    return left, right

