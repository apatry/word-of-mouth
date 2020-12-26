import unittest

from spacy.lang.en import English
from spacy.tokens import Doc

from wordofmouth.database import MusicBand
from wordofmouth.pipelines import (
    BandNameNerPipe,
    bands_nlp,
)

from peewee import SqliteDatabase

test_db = SqliteDatabase(":memory:")
TABLES = [MusicBand]


class BandNameNerTests(unittest.TestCase):
    def setUp(self):
        """
        Setup the testing environment.

        See
        http://docs.peewee-orm.com/en/latest/peewee/database.html#testing
        for more details about how to test with databases.
        """
        test_db.bind(TABLES)
        test_db.connect()
        test_db.create_tables(TABLES)

        # add mock data to the database
        MusicBand.create(
            name="Glass Animals", wikidata_id="https://www.wikidata.org/wiki/Q16839267",
        )

    def tearDown(self):
        test_db.drop_tables(TABLES)
        test_db.close()

    def test_bands_ner_pipeline(self):
        nlp = bands_nlp()
        doc = nlp("Glass Animals is one of my favorite band.")

        assert len(doc._.band_ents) == 1
        assert doc._.band_ents[0].text == "Glass Animals"
