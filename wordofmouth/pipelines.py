"""
Collection of NLP pipelines.
"""

from typing import Iterable, List, Optional

import spacy
from spacy.lang.en import English
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span

from .database import MusicBand


def bands_nlp() -> Language:
    """
    Create a NLP pipeline matching music band names in a document.

    The band entities are stored in ``doc._.band_ents``.
    """
    nlp = English()
    nlp.add_pipe(BandNameNerPipe(nlp, MusicBand.select(), "band_ents"))

    return nlp


class BandNameNerPipe(object):
    """
    A pipe detecting band names in a document.
    """

    def __init__(
        self,
        nlp: Language,
        bands: Iterable[MusicBand],
        destination: Optional[str] = None,
    ):
        """
        :param nlp: The nlp pipeline used to tokenize the phrases.
        :param bands: The bands to match against.
        :param destination: The document extension where the entities
            should be stored.  When it is missing, the entities are
            stored in ``doc.ents``.
        """
        docs = nlp.pipe((band.name for band in bands))
        self._matcher = PhraseMatcher(nlp.vocab)
        self._matcher.add("music_band", docs)

        if destination and not Doc.has_extension(destination):
            Doc.set_extension(destination, default=None)
        self._destination = destination

    def __call__(self, doc: Doc) -> Doc:
        """
        Extract band name entities.

        :param doc: The document from which to extract and where to
            store the entities.
        """
        entities = []

        # match the entities
        for (match_id, start, end) in self._matcher(doc):
            entities.append(Span(doc, start, end, label=match_id))

        self._store_entities(doc, entities)

        return doc

    def _store_entities(self, doc: Doc, entities: List[Span]) -> None:
        """
        Store entities in a document.

        When ``self._destination`` is ``None``, the entities are
        stored in ``doc.ents``.  When it is not ``None``, they are
        stored in a document extension named ``self._destination``.

        :param doc: The document where the entities are stored.
        :param entities: The entities to store.
        """
        # store the entities at the right place
        if self._destination:
            doc._.set(self._destination, entities)
        else:
            doc.ents = entities
