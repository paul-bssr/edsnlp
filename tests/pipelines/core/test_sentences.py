import pytest
from pytest import mark

import edsnlp

text = (
    "Le patient est admis pour des douleurs dans le bras droit. "
    "mais n'a pas de problème de locomotion. \n"
    "Historique d'AVC dans la famille\n"
    "Mais ne semble pas en être un\n"
    "Pourrait être un cas de rhume.\n"
    "Motif :\n"
    "Douleurs dans le bras droit !"
    "Il est contaminé à E.Coli? c'est un problème, il faut s'en occuper."
)


@mark.parametrize("endlines", [True, False])
def test_sentences(endlines):
    nlp = edsnlp.blank("fr")
    nlp.add_pipe("eds.sentences", config={"use_endlines": endlines})
    doc = nlp.make_doc(text)

    if endlines:
        doc[28].tag_ = "EXCLUDED"

    doc = nlp(doc)

    sents_text = [sent.text for sent in doc.sents]

    if endlines:
        assert sents_text == [
            "Le patient est admis pour des douleurs dans le bras droit.",
            "mais n'a pas de problème de locomotion. \n",
            "Historique d'AVC dans la famille\nMais ne semble pas en être un\n",
            "Pourrait être un cas de rhume.\n",
            "Motif :\n",
            "Douleurs dans le bras droit !",
            "Il est contaminé à E.Coli?",
            "c'est un problème, il faut s'en occuper.",
        ]
    else:
        assert sents_text == [
            "Le patient est admis pour des douleurs dans le bras droit.",
            "mais n'a pas de problème de locomotion. \n",
            "Historique d'AVC dans la famille\n",
            "Mais ne semble pas en être un\n",
            "Pourrait être un cas de rhume.\n",
            "Motif :\n",
            "Douleurs dans le bras droit !",
            "Il est contaminé à E.Coli?",
            "c'est un problème, il faut s'en occuper.",
        ]

    nlp("")


def test_false_positives(blank_nlp):
    false_positives = [
        "02.04.2018",
        "E.Coli",
    ]

    for fp in false_positives:
        doc = blank_nlp(fp)
        assert len(list(doc.sents)) == 1


def test_newlines_double():
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        "eds.sentences",
        config={
            "punct_chars": [],
            "ignore_excluded": False,
            "check_capitalized": False,
            "min_newline_count": 2,
        },
    )

    doc = nlp(
        """\
Lundi
Mardi
Mercredi
Le patient est admis. Des douleurs dans le bras droit
\n\n
jeudi."""
    )
    assert len(list(doc.sents)) == 2

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        "eds.sentences",
        config={
            "punct_chars": [],
            "ignore_excluded": False,
            "check_capitalized": True,
            "min_newline_count": 2,
        },
    )

    doc = nlp(
        """\
Lundi
Mardi
Mercredi
Le patient est admis. Des douleurs dans le bras droit
\n
jeudi."""
    )
    assert len(list(doc.sents)) == 1


def test_sentences_bullet_starters():
    """Test specific bullet starter functionality"""
    nlp = edsnlp.blank("fr")
    nlp.add_pipe(
        edsnlp.pipes.sentences(
            use_bullet_start=True,
            bullet_starters=[
                "-",
            ],
        )
    )

    text = (
        "Symptômes observés:\n"
        "- Douleur thoracique\n"
        "-- forte toux\n"
        "- Fièvre élevée\n"
        "- Toux sèche\n"
        "Le patient semble stable - pas d'évolution\n"
    )

    doc = nlp(text)
    sents_text = [sent.text for sent in doc.sents]

    assert sents_text == [
        "Symptômes observés:\n",
        "- Douleur thoracique\n-- forte toux\n",
        "- Fièvre élevée\n",
        "- Toux sèche\n",
        "Le patient semble stable - pas d'évolution\n",
    ]


def test_sentences_bullet_edge_cases():
    """Test edge cases for bullet starters"""
    nlp = edsnlp.blank("fr")
    nlp.add_pipe(
        edsnlp.pipes.sentences(
            use_bullet_start=True,
            bullet_starters=[
                "-",
            ],
        )
    )

    text1 = "Le patient - âgé de 45 ans - présente des symptômes."
    doc1 = nlp(text1)
    assert len(list(doc1.sents)) == 1

    text2 = "Symptômes:   \n- Fièvre\t\n- Toux"
    doc2 = nlp(text2)
    sents2 = [sent.text for sent in doc2.sents]
    print(sents2)
    assert sents2 == ["Symptômes:   \n", "- Fièvre\t\n", "- Toux"]

    text3 = "Item:\n_ Premier point\n_ Deuxième point"
    doc3 = nlp(text3)
    sents3 = [sent.text for sent in doc3.sents]
    assert len(sents3) == 1


def test_sentences_multiple_bullet_types():
    """Test multiple bullet starter types"""
    nlp = edsnlp.blank("fr")
    nlp.add_pipe(
        "eds.sentences",
        config={"use_bullet_start": True, "bullet_starters": ["-", "*", "•", "·"]},
    )

    text = "Liste mixte:\n- Point A\n* Point B\n• Point C\n· Point D"
    doc = nlp(text)
    assert len(list(doc.sents)) == 5  # header + 4 bullets


##########################################################
############### Test Capitalized shapes ##################
##########################################################


def make_nlp(
    cap_shapes=None,
    mode: str | None = None,
    check_capitalized: bool = True,
    use_bullet_start: bool = True,
):
    """
    Build a pipeline configured for sentence segmentation with optional
    capitalized shapes override and mode selection.
    """
    nlp = edsnlp.blank("fr")
    config = {
        "use_bullet_start": use_bullet_start,
        "bullet_starters": ["-"],
        "check_capitalized": check_capitalized,
        "capitalized_shapes": cap_shapes,  # None => resolved from mode in sentences.py
    }
    if mode is not None:
        config["capitalized_mode"] = mode  # "legacy" | "expanded"
    nlp.add_pipe("eds.sentences", config=config)
    return nlp


def test_all_caps_sections_expanded_mode():
    """
    Expanded preset should detect ALL-CAPS headers as separate sentences.
    """
    nlp = make_nlp(mode="expanded")
    doc = nlp("CONCLUSION\nSuite\n")
    assert [s.text for s in doc.sents] == ["CONCLUSION\n", "Suite\n"]


def test_all_caps_with_bullets_expanded_mode():
    """
    Header on one line + two bullets on following lines => 3 sentences.
    """
    nlp = make_nlp(mode="expanded")
    text = "EVOLUTION\n- Fièvre\n- Toux\n"
    assert [s.text for s in nlp(text).sents] == [
        "EVOLUTION\n",
        "- Fièvre\n",
        "- Toux\n",
    ]


def test_custom_shapes_override_titlecase_only():
    """
    If capitalized_shapes is provided, it overrides the mode preset.
    Only Title Case 'Xxxxx' should trigger, not ALL-CAPS.
    """
    nlp = make_nlp(cap_shapes=["Xxxxx"])  # override explicit
    doc = nlp("Titre\nSuite\n")
    sents = [s.text for s in doc.sents]
    assert sents == ["Titre\n", "Suite\n"]


def test_disable_capitalized_rule_keeps_bullets_only():
    """
    With capitalization disabled, bullets should still isolate sentences if enabled.
    """
    nlp = make_nlp(check_capitalized=False)
    text = "CONCLUSION\n- Fièvre\n- Toux\n"
    sents = [s.text for s in nlp(text).sents]
    assert "- Fièvre\n" in sents and "- Toux\n" in sents


@pytest.mark.parametrize(
    "text, expected",
    [
        # Mixed whitespace before newline
        ("ÉTAT CIVIL  \nSuite\n", ["ÉTAT CIVIL  \n", "Suite\n"]),
        # Windows CRLF handled too
        ("CONCLUSION\r\n- Fièvre\r\n", ["CONCLUSION\r\n", "- Fièvre\r\n"]),
    ],
)
def test_newline_robustness_with_expanded_mode(text, expected):
    """
    The Cython component should detect newlines even when the token contains mixed
    whitespace or CRLF. Expanded mode then allows ALL-CAPS headers to start a sentence.
    """
    nlp = make_nlp(mode="expanded")
    doc = nlp(text)
    assert [s.text for s in doc.sents] == expected


def test_legacy_mode_behavior_non_regression():
    """
    Legacy mode preserves historical shapes:
    ("X'", "Xx", "Xxx", "Xxxx", "Xxxxx")
    Depending on legacy baseline, ALL-CAPS may or may not split on newline.
    Adjust this assertion to your historical behavior.
    """
    nlp = make_nlp(mode="legacy")
    doc = nlp("hémoculture\n\nCONCLUSION\nSuite\n")
    sents = [s.text for s in doc.sents]
    assert sents == ["hémoculture\n\nCONCLUSION\n", "Suite\n"]
