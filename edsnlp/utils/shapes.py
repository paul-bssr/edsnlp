# edsnlp/utils/shapes.py
from typing import List, Tuple


def generate_capitalized_shapes(
    upper_min: int = 2,
    upper_max: int = 13,
    x_min: int = 2,
    x_max: int = 12,
    include_all_caps: bool = True,
    include_titlecase: bool = True,
    include_apostrophe: bool = True,
) -> Tuple[str, ...]:
    """
    Generate spaCy `token.shape_` patterns used to detect capitalized line starts,
    such as ALL CAPS section titles, short Title Case tokens, and apostrophe forms (X').

    Patterns:
      - ALL CAPS: "XX", ..., "XXXXXXXXXXXXX" (configurable length range)
      - Title Case (short): "Xx", "Xxx", ..., "Xxxxxxxxxxxx"
      - Apostrophe: "X'"

    Notes
    -----
    - spaCy `token.shape_` maps uppercase letters to "X", lowercase to "x",
      digits to "d", and keeps punctuation as-is (e.g., apostrophe stays "'").
    - Accented letters (e.g., É, À, ç) are treated as letters by spaCy and
      thus follow the same "X"/"x" rules, which suits French clinical text well.

    Examples
    --------
    >>> shapes = generate_capitalized_shapes(upper_min=2, upper_max=3, x_min=2, x_max=3)
    >>> "XX" in shapes and "XXX" in shapes
    True
    >>> "Xx" in shapes and "Xxx" in shapes
    True
    >>> "X'" in shapes
    True
    """
    shapes: List[str] = []

    if include_all_caps:
        for i in range(upper_min, upper_max + 1):
            shapes.append("X" * i)

    if include_titlecase:
        for i in range(x_min, x_max + 1):
            shapes.append("X" + "x" * (i - 1))

    if include_apostrophe:
        shapes.append("X'")

    return tuple(shapes)


DEFAULT_CAPITALIZED_SHAPES: Tuple[str, ...] = generate_capitalized_shapes(
    upper_min=2, upper_max=13, x_min=2, x_max=12, include_apostrophe=True
)

LEGACY_CAPITALIZED_SHAPES: Tuple[str, ...] = ("X'", "Xx", "Xxx", "Xxxx", "Xxxxx")

__all__ = [
    "generate_capitalized_shapes",
    "DEFAULT_CAPITALIZED_SHAPES",
    "LEGACY_CAPITALIZED_SHAPES",
]
