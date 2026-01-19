import warnings
from typing import List, Optional

from spacy.tokens import Doc

from edsnlp.core import PipelineProtocol
from edsnlp.utils.shapes import DEFAULT_CAPITALIZED_SHAPES, LEGACY_CAPITALIZED_SHAPES

from ...base import BaseComponent
from .fast_sentences import FastSentenceSegmenter
from .terms import punctuation


class SentenceSegmenter(BaseComponent):
    r'''
    The `eds.sentences` matcher provides an alternative to spaCy's default
    `sentencizer`, aiming to overcome some of its limitations.

    Indeed, the `sentencizer` merely looks at period characters to detect the end of a
    sentence, a strategy that often fails in a clinical note settings. Our
    `eds.sentences` component also classifies end-of-lines as sentence boundaries if
    the subsequent token begins with an uppercase character, leading to slightly better
    performances. It can additionally leverage capitalized section headers and
    bullet-like list starters, which are frequent in structured medical documents.

    Moreover, the `eds.sentences` component use the output of the `eds.normalizer`
    and `eds.endlines` output by default when these components are added to the
    pipeline.

    Examples
    --------
    === "EDS-NLP"

        ```{ .python .no-check }
        import edsnlp, edsnlp.pipes as eds

        nlp = edsnlp.blank("eds")
        nlp.add_pipe(eds.sentences())  # same as nlp.add_pipe("eds.sentences")

        text = """Le patient est admis le 23 août 2021 pour une douleur à l'estomac
        Il lui était arrivé la même chose il y a deux ans."
        """

        doc = nlp(text)

        for sentence in doc.sents:
            print("<s>", sentence, "</s>")
        # Out: <s> Le patient est admis le 23 août 2021 pour une douleur à l'estomac
        # Out:  <\s>
        # Out: <s> Il lui était arrivé la même chose il y a deux ans. <\s>
        ```

    === "spaCy sentencizer"

        ```{ .python .no-check }
        import edsnlp, edsnlp.pipes as eds

        nlp = edsnlp.blank("eds")
        nlp.add_pipe("sentencizer")

        text = """Le patient est admis le 23 août 2021 pour une douleur à l'estomac"
        Il lui était arrivé la même chose il y a deux ans.
        """

        doc = nlp(text)

        for sentence in doc.sents:
            print("<s>", sentence, "</s>")
        # Out: <s> Le patient est admis le 23 août 2021 pour une douleur à l'estomac
        # Out: Il lui était arrivé la même chose il y a deux ans. <\s>
        ```

    Notice how EDS-NLP's implementation is more robust to ill-defined sentence endings.


    Parameters
    ----------
    nlp: PipelineProtocol
        The EDS-NLP pipeline
    name: Optional[str]
        The name of the component
    punct_chars: Optional[List[str]]
        Punctuation characters.
    use_endlines: bool
        Whether to use endlines prediction.
    ignore_excluded: bool
        Whether to ignore excluded tokens.
    check_capitalized: bool
        Whether to check for capitalized words after newlines or full stops.
    capitalized_mode : Optional[str], {"legacy", "expanded"}, default "legacy"
        Selects the preset of capitalized shapes used when `check_capitalized=True`
        and no explicit `capitalized_shapes` are provided.
        - "legacy": historical set of shapes ("X'", "Xx", "Xxx", "Xxxx", "Xxxxx",).
        - "expanded": extended set including ALL-CAPS and long Title Case patterns,
          improving detection of section headers.
    capitalized_shapes: Optional[List[str]]
        Capitalized shapes.
    min_newline_count: int
        The minimum number of newlines to consider a newline-triggered sentence.
    use_bullet_start: bool
        Whether to check for bullet starters after newlines or full stops.
    bullet_starters: Optional[List[str]]
        Bullet starters characters.

    Authors and citation
    --------------------
    The `eds.sentences` component was developed by AP-HP's Data Science team.
    '''

    def __init__(
        self,
        nlp: PipelineProtocol,
        name: Optional[str] = "sentences",
        punct_chars: Optional[List[str]] = None,
        use_endlines: Optional[bool] = None,
        ignore_excluded: bool = True,
        check_capitalized: bool = True,
        capitalized_mode: Optional[str] = "legacy",
        capitalized_shapes: Optional[List[str]] = None,
        min_newline_count: int = 1,
        use_bullet_start: bool = False,
        bullet_starters: Optional[List[str]] = None,
    ):
        super().__init__(nlp, name)
        if min_newline_count > 1 and nlp.lang != "eds":
            warnings.warn(
                "To use min_newline_count > 1, you need to use the 'eds' language "
                "in order to split newlines into separate and countable tokens."
            )

        if punct_chars is None:
            punct_chars = punctuation

        if check_capitalized:
            if capitalized_shapes is not None:
                capitalized_shapes = tuple(capitalized_shapes)
            else:
                capitalized_shapes = (
                    LEGACY_CAPITALIZED_SHAPES
                    if capitalized_mode == "legacy"
                    else DEFAULT_CAPITALIZED_SHAPES
                )
        else:
            capitalized_shapes = ()

        self.fast_segmenter = FastSentenceSegmenter(
            vocab=nlp.vocab,
            punct_chars=punct_chars,
            use_endlines=use_endlines,
            ignore_excluded=ignore_excluded,
            check_capitalized=check_capitalized,
            capitalized_shapes=capitalized_shapes,
            min_newline_count=min_newline_count,
            use_bullet_start=use_bullet_start,
            bullet_starters=bullet_starters,
        )

    def __call__(self, doc: Doc):
        return self.fast_segmenter(doc)
