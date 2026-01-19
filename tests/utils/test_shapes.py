from edsnlp.utils.shapes import (
    DEFAULT_CAPITALIZED_SHAPES,
    LEGACY_CAPITALIZED_SHAPES,
    generate_capitalized_shapes,
)


def test_generate_returns_tuple_of_strings_and_no_duplicates():
    shapes = generate_capitalized_shapes()
    assert isinstance(shapes, tuple)
    assert all(isinstance(s, str) for s in shapes)
    # no duplicates
    assert len(shapes) == len(set(shapes))


def test_toggles_all_caps_titlecase_apostrophe():
    # all ON
    s_all = generate_capitalized_shapes(
        include_all_caps=True, include_titlecase=True, include_apostrophe=True
    )
    assert "XX" in s_all and "Xx" in s_all and "X'" in s_all

    # all caps OFF
    s_no_caps = generate_capitalized_shapes(
        include_all_caps=False, include_titlecase=True, include_apostrophe=True
    )
    assert "XX" not in s_no_caps and "Xx" in s_no_caps and "X'" in s_no_caps

    # titlecase OFF
    s_no_title = generate_capitalized_shapes(
        include_all_caps=True, include_titlecase=False, include_apostrophe=True
    )
    assert "XX" in s_no_title and "Xx" not in s_no_title and "X'" in s_no_title

    # apostrophe OFF
    s_no_apo = generate_capitalized_shapes(
        include_all_caps=True, include_titlecase=True, include_apostrophe=False
    )
    assert "XX" in s_no_apo and "Xx" in s_no_apo and "X'" not in s_no_apo

    # all OFF
    s_none = generate_capitalized_shapes(
        include_all_caps=False, include_titlecase=False, include_apostrophe=False
    )
    assert s_none == tuple()


def test_defaults_match_default_constant():
    expected = generate_capitalized_shapes(
        upper_min=2, upper_max=13, x_min=2, x_max=12, include_apostrophe=True
    )
    assert DEFAULT_CAPITALIZED_SHAPES == expected


def test_legacy_exact_values():
    assert LEGACY_CAPITALIZED_SHAPES == ("X'", "Xx", "Xxx", "Xxxx", "Xxxxx")


def test_bounds_min_max_presence():
    s = generate_capitalized_shapes(
        upper_min=2, upper_max=4, x_min=2, x_max=3, include_apostrophe=False
    )
    assert "XX" in s and "XXXX" in s
    assert "Xx" in s and "Xxx" in s
