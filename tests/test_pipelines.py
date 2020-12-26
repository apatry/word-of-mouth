import unittest

from spacy.lang.en import English
from spacy.tokens import Doc

from wordofmouth.database import MusicBand
from wordofmouth.pipelines import (
    BandNameNerPipe,
    bands_nlp,
)


class BandNameNerTests(unittest.TestCase):
    def test_band_name_ner_pipe(self):
        nlp = English()
        bands = [
            MusicBand(
                name="Glass Animals",
                wikidata_id="https://www.wikidata.org/wiki/Q16839267",
            )
        ]

        ner = BandNameNerPipe(nlp, bands)
        doc = ner(nlp.make_doc("Glass Animals is one of my favorite band."))
        assert len(doc.ents) == 1
        assert doc.ents[0].text == "Glass Animals"

    def test_bands_ner_pipeline(self):
        nlp = bands_nlp()
        doc = nlp("Glass Animals is one of my favorite band.")

        # TODO Run real test against a mocked database.
        print(f"ents: {doc.ents}")
        print(f"bands: {doc._.band_ents}")
