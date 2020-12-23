"""
Everything that is needed to add or extract data from our database.
"""

from peewee import CharField, DateTimeField, Model, SqliteDatabase  # type: ignore

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

    wikidata_id = CharField(unique=True, help_text="The entity id in wikidata.")
    name = CharField(help_text="The English name of the band.")

    class Meta:
        database = db


class RedditPost(Model):
    """
    A reddit post.

    We only keep the following field:

    - created_at: The creation time of the post (maps to created_utc in pushshift).
    - permalink: Permalink to the post (maps to permalink in pushshift).
    - flair: The flair text (maps to link_flair_text in pushshift).
    - title: The title of the post (maps to title in pushshift).
    - selftext: The text of the post (maps to selftext in pushshift).
    """

    created_at = DateTimeField(help_text="The creation time of the post.")
    permalink = CharField(unique=True, help_text="A link to the post.")
    flair = CharField(null=True, help_text="Flair text when present.")
    title = CharField(help_text="The post title.")
    selftext = CharField(null=True, help_text="The post text.")

    class Meta:
        database = db


def create_tables() -> None:
    """
    Create all the tables needed by the project.
    """
    db.create_tables([MusicBand, RedditPost])
