from datetime import datetime
from typing import Dict, Optional

from peewee import chunked  # type: ignore
from wordofmouth.database import MusicBand, RedditPost
from wordofmouth.etl.reddit import crawl_subreddit
from wordofmouth.etl.sparql import query_wikidata

_PEEWEE_CHUNK_SIZE = 512


def etl_music_bands(db, *, chunk_size: int = 512, limit: Optional[int] = None) -> int:
    """
    ETL music bands into our databases.

    :param db: The database in which to ETL the bands.
    :param chunk_size: The number of records in each insert batch.
    :param limit: The maximum number of records to fetch from
        wikidata.  This is included for debugging purpose.

    :return: The number of records that were ETLed.
    """

    query = """
    SELECT DISTINCT ?wikidata_id ?name
    WHERE
    {
      ?wikidata_id wdt:P31/wdt:P279* wd:Q2088357.
      ?wikidata_id rdfs:label ?name.
      FILTER (LANG(?name) = 'en')
    }
    """

    if limit:
        query = f"{query}\nLIMIT {limit}"

    bands = query_wikidata(query)
    with db.atomic():
        for batch in chunked(bands, _PEEWEE_CHUNK_SIZE):
            MusicBand.insert_many(batch).execute()

    return len(bands)


def etl_reddit_posts(db, max_posts=2000, *, chunk_size=512) -> int:
    """
    ETL posts from r/ifyoulikeblank on reddit.

    :param db: The database in which to ETL the posts.
    :param max_posts: The maximum number of posts to ETL.

    :return: The number of posts that were downloaded.
    """
    posts = [
        {
            "created_at": datetime.fromtimestamp(post["created_utc"]),
            "permalink": post["permalink"],
            "flair": post.get("link_flair_text", None),
            "title": post["title"],
            "selftext": post.get("selftext", None),
        }
        for post in crawl_subreddit("ifyoulikeblank", max_posts=max_posts)
    ]

    with db.atomic():
        for batch in chunked(posts, _PEEWEE_CHUNK_SIZE):
            RedditPost.insert_many(batch).execute()

    return len(posts)
