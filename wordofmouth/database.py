"""
Everything that is needed to add or extract data from our database.
"""

from peewee import CharField, Model, SqliteDatabase  # type: ignore

# TODO this should be configured outside of this module. I will keep
# it hardcoded until I have a better hold of how to use peewee as my
# orm.
db = SqliteDatabase("wordofmouth.db")


class MusicBand(Model):
    """
    A music band has a name and a unique wikidata identifier.

    At this moment, we assume that we can only get band name from
    wikidata.  If it ever changes, we will update the schema to use
    another kind of unique identifier.
    """

    wikidata_id = CharField(unique=True)
    name = CharField()

    class Meta:
        database = db


def create_tables() -> None:
    """
    Create all the tables needed by the project.
    """
    db.create_tables([MusicBand])
